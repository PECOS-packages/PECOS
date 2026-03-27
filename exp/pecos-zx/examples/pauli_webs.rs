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

//! Pauli web computation on a simple QEC circuit.
//!
//! Demonstrates:
//! - Building a repetition code syndrome extraction circuit
//! - Converting to ZX
//! - Computing Pauli webs
//! - Classifying webs as detectors/stabilizers
//! - Rendering with web overlay

use pecos_quantum::DagCircuit;
use pecos_zx::convert::dag_to_zx;
use pecos_zx::pauli_web::{WebClassification, classify_webs, compute_pauli_webs};
use pecos_zx::viz::{Renderer, SvgOptions, WebOverlay, render_html_with_rewrites};

fn main() {
    println!("=== Pauli Webs ===\n");

    // Build a repetition code with 2 rounds of syndrome extraction:
    //   data qubits: 0, 1, 2
    //   round 1 ancillas: 3, 4
    //   round 2 ancillas: 5, 6
    //
    // Two rounds are needed so that detection_webs can find non-trivial
    // nullspace vectors. With only one round, every data-qubit spider is
    // boundary-adjacent and the no_output constraint zeros them all out.
    // Fresh qubit indices are used for round 2 because QuiZX's Measure
    // removes the qubit from the active map.
    let mut dag = DagCircuit::new();

    // --- Round 1 ---
    dag.pz(&[3]);
    dag.pz(&[4]);

    // Check Z0*Z1 via ancilla 3
    dag.cx(&[(0, 3)]);
    dag.cx(&[(1, 3)]);
    // Check Z1*Z2 via ancilla 4
    dag.cx(&[(1, 4)]);
    dag.cx(&[(2, 4)]);

    dag.mz(&[3]);
    dag.mz(&[4]);

    // --- Round 2 (fresh ancilla indices) ---
    dag.pz(&[5]);
    dag.pz(&[6]);

    // Check Z0*Z1 via ancilla 5
    dag.cx(&[(0, 5)]);
    dag.cx(&[(1, 5)]);
    // Check Z1*Z2 via ancilla 6
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
    r.render(&graph, "pauli_webs_circuit");

    // Compute Pauli webs
    let result = compute_pauli_webs(&graph);
    println!("\nPauli web computation:");
    println!("  Found {} webs", result.webs.len());

    // Classify webs
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

    println!("  Detectors: {num_detectors}");
    println!("  Input stabilizers: {num_input_stab}");
    println!("  Output stabilizers: {num_output_stab}");
    println!("  Propagated: {num_propagated}\n");

    // Render one SVG per web for clarity
    let overlay = WebOverlay::from_result(&result);
    for i in 0..overlay.len() {
        r.svg.web_overlay = Some(overlay.single(i));
        r.render(&graph, &format!("pauli_webs_web_{i}"));
    }

    // Also render combined overlay
    r.svg.web_overlay = Some(overlay.clone());
    r.render(&graph, "pauli_webs_overlay");

    // Render interactive HTML viewer with rewrite exploration
    let html_opts = SvgOptions {
        web_overlay: Some(overlay),
        ..SvgOptions::default()
    };
    let html = render_html_with_rewrites(&graph, &html_opts);
    let html_path = std::path::Path::new("exp/pecos-zx/examples/output").join("pauli_webs.html");
    std::fs::write(&html_path, &html).expect("failed to write HTML");
    println!("  Wrote {}", html_path.display());
}
