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

//! Full DEM extraction pipeline on a repetition code.
//!
//! Demonstrates:
//! - Building a 2-round repetition code syndrome extraction circuit
//! - Converting to ZX
//! - Computing and classifying Pauli webs
//! - Applying uniform depolarizing noise
//! - Extracting a Detector Error Model (DEM)
//! - Printing the DEM in Stim format

use pecos_quantum::DagCircuit;
use pecos_zx::convert::dag_to_zx;
use pecos_zx::dem::Dem;
use pecos_zx::noise::NoiseModel;
use pecos_zx::pauli_web::{WebClassification, classify_webs, compute_pauli_webs};
use pecos_zx::viz::Renderer;

fn main() {
    println!("=== DEM Extraction Pipeline ===\n");

    // Build a repetition code with 2 rounds of syndrome extraction:
    //   data qubits: 0, 1, 2
    //   round 1 ancillas: 3, 4
    //   round 2 ancillas: 5, 6
    let mut dag = DagCircuit::new();

    // --- Round 1 ---
    dag.pz(&[3]);
    dag.pz(&[4]);
    dag.cx(&[(0, 3)]);
    dag.cx(&[(1, 3)]);
    dag.cx(&[(1, 4)]);
    dag.cx(&[(2, 4)]);
    dag.mz(&[3]);
    dag.mz(&[4]);

    // --- Round 2 (fresh ancilla indices) ---
    dag.pz(&[5]);
    dag.pz(&[6]);
    dag.cx(&[(0, 5)]);
    dag.cx(&[(1, 5)]);
    dag.cx(&[(1, 6)]);
    dag.cx(&[(2, 6)]);
    dag.mz(&[5]);
    dag.mz(&[6]);

    println!("Repetition code (2-round syndrome extraction):");
    println!("  Data qubits: 0, 1, 2");
    println!("  Round 1 ancillas: 3, 4");
    println!("  Round 2 ancillas: 5, 6");
    println!("  Gates: {}\n", dag.gate_count());

    // Convert to ZX
    let graph = dag_to_zx(&dag).expect("conversion failed");

    // Render the base circuit
    let mut r = Renderer::default();
    r.set_output_dir("exp/pecos-zx/examples/output");
    r.render(&graph, "dem_circuit");

    // Compute Pauli webs
    let result = compute_pauli_webs(&graph);
    let classifications = classify_webs(&result);

    let num_detectors = classifications
        .iter()
        .filter(|c| **c == WebClassification::Detector)
        .count();
    let num_input_stab = classifications
        .iter()
        .filter(|c| **c == WebClassification::InputStabilizer)
        .count();
    let num_output_stab = classifications
        .iter()
        .filter(|c| **c == WebClassification::OutputStabilizer)
        .count();
    let num_propagated = classifications
        .iter()
        .filter(|c| **c == WebClassification::Propagated)
        .count();

    println!("Pauli web classification:");
    println!("  Detectors:          {num_detectors}");
    println!("  Input stabilizers:  {num_input_stab}");
    println!("  Output stabilizers: {num_output_stab}");
    println!("  Propagated:         {num_propagated}\n");

    // Apply uniform depolarizing noise
    let p = 0.001;
    let noise = NoiseModel::uniform_depolarizing(&graph, p);
    println!("Noise model:");
    println!("  Depolarizing rate: {p}");
    println!("  Noisy edges: {}\n", noise.edge_errors.len());

    // Build DEM
    let dem = Dem::from_webs(&result, &noise);
    println!("DEM statistics:");
    println!("  Detectors:   {}", dem.detectors.len());
    println!("  Observables: {}", dem.observables.len());
    println!("  Errors:      {}\n", dem.errors.len());

    // Print in Stim format
    println!("--- Stim DEM ---");
    println!("{}", dem.to_stim_string());
}
