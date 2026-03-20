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

//! TikZ output for ZX diagrams.
//!
//! Generates LaTeX/TikZ code suitable for inclusion in papers and documents.
//! Demonstrates:
//! 1. Bell state circuit as a standalone LaTeX document
//! 2. Graph state as a TikZ snippet (for embedding in existing documents)
//! 3. Before/after simplification comparison

use pecos_quantum::DagCircuit;
use pecos_zx::convert::dag_to_zx;
use pecos_zx::graph::from_adjacency_matrix;
use pecos_zx::simplify;
use pecos_zx::viz::Renderer;
use pecos_zx::viz::tikz::{TikzOptions, render_tikz};

fn main() {
    println!("=== TikZ Output for ZX Diagrams ===\n");

    bell_state_tikz();
    graph_state_tikz();
    simplification_tikz();

    println!("\nCompile the .tex files with pdflatex to produce PDFs.");
    println!("The .tikz files can be \\input{{}} into existing LaTeX documents");
    println!("(add the preamble from tikz_preamble() to your document).");
}

/// Bell state circuit -> ZX -> all formats.
fn bell_state_tikz() {
    let mut dag = DagCircuit::new();
    dag.h(0);
    dag.cx(0, 1);

    let graph = dag_to_zx(&dag).expect("conversion failed");

    let mut r = Renderer::default();
    r.set_output_dir("exp/pecos-zx/examples/output");
    r.render(&graph, "bell_state");
}

/// Graph state: raw TikZ snippet for embedding, plus full render.
fn graph_state_tikz() {
    // Linear cluster state: 0-1-2-3
    #[rustfmt::skip]
    let linear = vec![
        false, true,  false, false,
        true,  false, true,  false,
        false, true,  false, true,
        false, false, true,  false,
    ];
    let graph = from_adjacency_matrix(&linear, 4);

    // Raw TikZ snippet (no environment wrapper, no preamble) -- special case
    let opts = TikzOptions {
        wrap_in_environment: false,
        show_boundary_labels: false,
        ..TikzOptions::default()
    };
    let tikz = render_tikz(&graph, &opts);
    let tikz_path =
        std::path::Path::new("exp/pecos-zx/examples/output").join("graph_state_linear.tikz");
    std::fs::create_dir_all("exp/pecos-zx/examples/output")
        .expect("failed to create output directory");
    std::fs::write(&tikz_path, &tikz).expect("failed to write .tikz");
    println!("  Wrote {} (raw TikZ snippet)", tikz_path.display());

    // Full render (ASCII + SVG + TikZ standalone)
    let mut r = Renderer::default();
    r.set_output_dir("exp/pecos-zx/examples/output");
    r.render(&graph, "graph_state_linear");
}

/// Before/after simplification in all formats.
fn simplification_tikz() {
    // Build a circuit with redundant gates: H*H = I, CX*CX = I
    let mut dag = DagCircuit::new();
    dag.h(0);
    dag.h(0);
    dag.cx(0, 1);
    dag.cx(0, 1);
    dag.h(1);

    let mut graph = dag_to_zx(&dag).expect("conversion failed");

    let mut r = Renderer::default();
    r.set_output_dir("exp/pecos-zx/examples/output");
    r.render(&graph, "simplify_before");

    // Simplify
    simplify::clifford_simp(&mut graph);

    println!();
    r.render(&graph, "simplify_after");
}
