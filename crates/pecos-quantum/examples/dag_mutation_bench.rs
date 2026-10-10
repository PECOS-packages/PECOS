// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Dependency-free removal benchmark. Run with `cargo run --release -p
//! pecos-quantum --example dag_mutation_bench -- [workload] [size] [qubits]`.
//! Construction and cloning are outside the timed region; reports three-run medians.
use pecos_core::{Angle64, Gate};
use pecos_quantum::DagCircuit;
use pecos_quantum::pass::{CancelInverses, CircuitPass, SimplifyRotations, StripIdentities};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let workload = args.get(1).map_or("remove", String::as_str);
    let n: usize = args.get(2).map_or(100_000, |s| s.parse().unwrap());
    let qubits: usize = args.get(3).map_or(10, |s| s.parse().unwrap());
    let mut base = DagCircuit::with_capacity(n, n * 2);
    for i in 0..n {
        let q = i % qubits;
        match workload {
            "strip" => {
                base.add_gate_auto_wire(if (i / qubits).is_multiple_of(2) {
                    Gate::i(&[q])
                } else {
                    Gate::h(&[q])
                });
            }
            "simplify" => {
                base.rzz(Angle64::HALF_TURN, &[(q, (q + 1) % qubits)]);
            }
            "alternate" => {
                base.h(&[0]);
            }
            _ => {
                base.h(&[q]);
            }
        }
    }
    let mut times = Vec::new();
    for _ in 0..3 {
        let mut circuit = base.clone();
        let start = Instant::now();
        match workload {
            "cancel" => CancelInverses.apply_dag(&mut circuit),
            "strip" => StripIdentities.apply_dag(&mut circuit),
            "simplify" => SimplifyRotations.apply_dag(&mut circuit),
            "remove" | "alternate" => {
                let stride = if workload == "alternate" { 2 } else { 1 };
                for node in (0..n).step_by(stride) {
                    black_box(circuit.remove_gate(node));
                }
            }
            _ => panic!("unknown workload: {workload}"),
        }
        times.push(start.elapsed());
        black_box(circuit);
    }
    times.sort_unstable();
    println!("{workload},{n},{qubits},{:.6}", times[1].as_secs_f64());
}
