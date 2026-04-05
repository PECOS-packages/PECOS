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

//! Analyze measurement patterns in surface code simulation.
//!
//! Run with:
//!   cargo run --release --example `analyze_measurements` -p pecos-simulators

use pecos_core::QubitId;
use pecos_simulators::{CliffordGateable, SparseStab};

struct SurfaceCodeParams {
    distance: usize,
    num_qubits: usize,
    num_data: usize,
    num_ancillas: usize,
    ancilla_start: usize,
}

impl SurfaceCodeParams {
    fn new(distance: usize) -> Self {
        let num_data = distance * distance;
        let num_ancillas = num_data - 1;
        let num_qubits = num_data + num_ancillas;
        Self {
            distance,
            num_qubits,
            num_data,
            num_ancillas,
            ancilla_start: num_data,
        }
    }

    fn ancilla_neighbors(&self, ancilla_idx: usize) -> Vec<usize> {
        let d = self.distance;
        let mut neighbors = Vec::with_capacity(4);
        let base = ancilla_idx % self.num_data;

        neighbors.push(base);
        if ancilla_idx + 1 < self.num_data {
            neighbors.push((base + 1) % self.num_data);
        }

        if ancilla_idx < self.num_ancillas / 2 {
            if base + d < self.num_data {
                neighbors.push(base + d);
            }
            if ancilla_idx > d && base >= d {
                neighbors.push(base - d);
            }
        } else {
            if base + d < self.num_data {
                neighbors.push(base + d);
            }
            if base + d + 1 < self.num_data {
                neighbors.push((base + d + 1) % self.num_data);
            }
        }

        neighbors
    }
}

fn analyze_measurements(params: &SurfaceCodeParams, rounds: usize) -> (usize, usize, usize) {
    let mut sim = SparseStab::new(params.num_qubits);
    let mut total_measurements = 0;
    let mut deterministic = 0;
    let mut nondeterministic = 0;

    // Initialize data qubits in |+> state
    for i in 0..params.num_data {
        sim.h(&[QubitId(i)]);
    }

    // Syndrome extraction rounds
    for round in 0..rounds {
        // CX gates
        for a in 0..params.num_ancillas {
            let ancilla = QubitId(params.ancilla_start + a);
            let neighbors = params.ancilla_neighbors(a);

            if a < params.num_ancillas / 2 {
                for &data in &neighbors {
                    sim.cx(&[(ancilla, QubitId(data))]);
                }
            } else {
                for &data in &neighbors {
                    sim.cx(&[(QubitId(data), ancilla)]);
                }
            }
        }

        // Measure all ancillas and track determinism
        let mut round_det = 0;
        let mut round_nondet = 0;
        for a in 0..params.num_ancillas {
            let ancilla = QubitId(params.ancilla_start + a);
            let results = sim.mz(&[ancilla]);
            total_measurements += 1;
            if results[0].is_deterministic {
                deterministic += 1;
                round_det += 1;
            } else {
                nondeterministic += 1;
                round_nondet += 1;
            }
        }

        println!(
            "  Round {}: {} deterministic, {} non-deterministic",
            round + 1,
            round_det,
            round_nondet
        );
    }

    (total_measurements, deterministic, nondeterministic)
}

#[allow(clippy::cast_precision_loss)] // percentage calculation
fn main() {
    println!("=== Surface Code Measurement Analysis ===\n");

    for distance in [5, 11, 17] {
        let params = SurfaceCodeParams::new(distance);
        let rounds = 5;

        println!(
            "Distance {} ({} qubits, {} ancillas, {} rounds):",
            distance, params.num_qubits, params.num_ancillas, rounds
        );

        let (total, det, nondet) = analyze_measurements(&params, rounds);

        println!(
            "\n  Total: {} measurements, {} deterministic ({:.1}%), {} non-deterministic ({:.1}%)\n",
            total,
            det,
            100.0 * det as f64 / total as f64,
            nondet,
            100.0 * nondet as f64 / total as f64
        );
    }

    println!("\n=== Conclusion ===");
    println!("If most measurements are non-deterministic, RNG batching could help.");
    println!("If most are deterministic, RNG is not a bottleneck.");
}
