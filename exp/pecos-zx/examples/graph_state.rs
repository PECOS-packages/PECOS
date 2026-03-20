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

//! Graph states: adjacency matrix -> ZX diagram -> render all formats.
//!
//! Demonstrates building graph states from adjacency matrices
//! (linear cluster, star, ring) and rendering them.

use pecos_zx::graph::{from_adjacency_matrix, graph_stats};
use pecos_zx::viz::Renderer;
use pecos_zx::viz::layout::LayoutAlgorithm;

fn main() {
    println!("=== Graph States ===\n");

    // Use force-directed layout so the graph topology is visible.
    // The default FromGraph layout places all Z-spiders in a single column,
    // making different topologies look nearly identical.
    let mut r = Renderer::default();
    r.set_output_dir("exp/pecos-zx/examples/output");
    r.set_layout(LayoutAlgorithm::ForceDirected);

    // Linear cluster state: 0-1-2-3
    #[rustfmt::skip]
    let linear = vec![
        false, true,  false, false,
        true,  false, true,  false,
        false, true,  false, true,
        false, false, true,  false,
    ];
    let g_linear = from_adjacency_matrix(&linear, 4);
    let stats = graph_stats(&g_linear);
    println!("Linear cluster (4 qubits):");
    println!(
        "  {} vertices, {} edges\n",
        stats.num_vertices, stats.num_edges
    );
    r.render(&g_linear, "graph_state_linear");

    // Star state: center=0 connected to 1,2,3,4
    #[rustfmt::skip]
    let star = vec![
        false, true,  true,  true,  true,
        true,  false, false, false, false,
        true,  false, false, false, false,
        true,  false, false, false, false,
        true,  false, false, false, false,
    ];
    let g_star = from_adjacency_matrix(&star, 5);
    let stats = graph_stats(&g_star);
    println!("\nStar graph (5 qubits):");
    println!(
        "  {} vertices, {} edges\n",
        stats.num_vertices, stats.num_edges
    );
    r.render(&g_star, "graph_state_star");

    // Ring state: 0-1-2-3-0
    #[rustfmt::skip]
    let ring = vec![
        false, true,  false, true,
        true,  false, true,  false,
        false, true,  false, true,
        true,  false, true,  false,
    ];
    let g_ring = from_adjacency_matrix(&ring, 4);
    let stats = graph_stats(&g_ring);
    println!("\nRing graph (4 qubits):");
    println!(
        "  {} vertices, {} edges\n",
        stats.num_vertices, stats.num_edges
    );
    r.render(&g_ring, "graph_state_ring");
}
