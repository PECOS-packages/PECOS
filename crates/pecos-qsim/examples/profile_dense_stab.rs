// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Profiling binary for DenseStab.
//!
//! Run with:
//!   cargo build --release --example profile_dense_stab -p pecos-qsim
//!   perf record -g ./target/release/examples/profile_dense_stab
//!   perf report

use pecos_core::QubitId;
use pecos_qsim::{CliffordGateable, DenseStab};
use std::hint::black_box;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let distance: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let iterations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10000);

    let num_data = distance * distance;
    let num_ancillas = num_data - 1;
    let num_qubits = num_data + num_ancillas;
    let ancilla_start = num_data;

    eprintln!("Profiling DenseStab");
    eprintln!("  Distance: {}", distance);
    eprintln!("  Qubits: {} ({} data + {} ancilla)", num_qubits, num_data, num_ancillas);
    eprintln!("  Rounds: {}", rounds);
    eprintln!("  Iterations: {}", iterations);

    for _ in 0..iterations {
        let mut sim = DenseStab::new(num_qubits);

        // Initialize data qubits in |+> state
        for i in 0..num_data {
            sim.h(&[QubitId(i)]);
        }

        // Syndrome extraction rounds
        for _round in 0..rounds {
            // CX gates for syndrome extraction
            for a in 0..num_ancillas {
                let ancilla = QubitId(ancilla_start + a);
                let base = a % num_data;

                // Simplified neighbor pattern
                if a < num_ancillas / 2 {
                    // X-type stabilizers
                    sim.cx(&[ancilla, QubitId(base)]);
                    if base + 1 < num_data {
                        sim.cx(&[ancilla, QubitId(base + 1)]);
                    }
                    if base + distance < num_data {
                        sim.cx(&[ancilla, QubitId(base + distance)]);
                    }
                } else {
                    // Z-type stabilizers
                    sim.cx(&[QubitId(base), ancilla]);
                    if base + 1 < num_data {
                        sim.cx(&[QubitId(base + 1), ancilla]);
                    }
                    if base + distance < num_data {
                        sim.cx(&[QubitId(base + distance), ancilla]);
                    }
                }
            }

            // Measure all ancillas
            for a in 0..num_ancillas {
                let ancilla = QubitId(ancilla_start + a);
                black_box(sim.mz(&[ancilla]));
            }
        }
    }

    eprintln!("Done.");
}
