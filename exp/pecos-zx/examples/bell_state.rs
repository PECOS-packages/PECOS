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

//! Bell state circuit -> ZX -> simplify -> render all formats
//!
//! Demonstrates the core pipeline:
//! 1. Build a Bell state circuit in PECOS
//! 2. Convert to a ZX graph
//! 3. Render the ZX graph (ASCII + SVG + TikZ)
//! 4. Apply Clifford simplification
//! 5. Render the simplified graph

use pecos_quantum::DagCircuit;
use pecos_zx::convert::dag_to_zx;
use pecos_zx::graph::graph_stats;
use pecos_zx::simplify;
use pecos_zx::viz::Renderer;

fn main() {
    // 1. Build a Bell state circuit: H(0), CX(0,1)
    let mut dag = DagCircuit::new();
    dag.h(0);
    dag.cx(0, 1);

    println!("=== Bell State Circuit -> ZX ===\n");

    // 2. Convert to ZX graph
    let mut graph = dag_to_zx(&dag).expect("conversion failed");

    let stats = graph_stats(&graph);
    println!("Before simplification:");
    println!(
        "  Vertices: {} ({} Z-spiders, {} X-spiders, {} boundaries)",
        stats.num_vertices, stats.num_z_spiders, stats.num_x_spiders, stats.num_boundaries
    );
    println!("  Edges: {}", stats.num_edges);
    println!(
        "  I/O: {} inputs, {} outputs\n",
        stats.num_inputs, stats.num_outputs
    );

    // 3. Render before simplification
    let mut r = Renderer::default();
    r.set_output_dir("exp/pecos-zx/examples/output");
    r.render(&graph, "bell_before");

    // 4. Apply Clifford simplification
    simplify::clifford_simp(&mut graph);

    let stats_after = graph_stats(&graph);
    println!("\nAfter clifford_simp:");
    println!(
        "  Vertices: {} ({} Z-spiders, {} X-spiders, {} boundaries)",
        stats_after.num_vertices,
        stats_after.num_z_spiders,
        stats_after.num_x_spiders,
        stats_after.num_boundaries
    );
    println!("  Edges: {}\n", stats_after.num_edges);

    // 5. Render after simplification
    r.render(&graph, "bell_after");
}
