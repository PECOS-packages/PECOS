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

//! Graph state analysis: construction, manipulation, and entanglement.
//!
//! Demonstrates the `GraphState` type for building graph states,
//! performing local complementation and vertex deletion, computing
//! entanglement measures, and converting to ZX diagrams.

use pecos_zx::graph::graph_stats;
use pecos_zx::graph_state::GraphState;

fn main() {
    println!("=== Graph State Analysis ===\n");

    // -- Construction --------------------------------------------------------

    // Linear cluster: 0-1-2-3
    let linear = {
        let mut gs = GraphState::empty(4);
        gs.toggle_edge(0, 1);
        gs.toggle_edge(1, 2);
        gs.toggle_edge(2, 3);
        gs
    };
    println!("Linear cluster (4 qubits):");
    println!("  edges: {}", linear.num_edges());
    println!(
        "  degree sequence: {:?}",
        (0..4).map(|v| linear.degree(v)).collect::<Vec<_>>()
    );
    println!("  neighbors of 1: {:?}", linear.neighbors(1));

    // Star: center=0 connected to 1,2,3
    let star = {
        let mut gs = GraphState::empty(4);
        for i in 1..4 {
            gs.toggle_edge(0, i);
        }
        gs
    };
    println!("\nStar graph (4 qubits):");
    println!("  edges: {}", star.num_edges());
    println!("  center degree: {}", star.degree(0));

    // Ring: 0-1-2-3-0
    let ring = {
        let mut gs = GraphState::empty(4);
        gs.toggle_edge(0, 1);
        gs.toggle_edge(1, 2);
        gs.toggle_edge(2, 3);
        gs.toggle_edge(3, 0);
        gs
    };
    println!("\nRing graph (4 qubits):");
    println!("  edges: {}", ring.num_edges());

    // -- Local complementation -----------------------------------------------

    println!("\n=== Local Complementation ===\n");

    let mut gs = star.clone();
    println!("Star before LC on center:");
    println!(
        "  edges among leaves (1-2, 1-3, 2-3): {}, {}, {}",
        gs.has_edge(1, 2),
        gs.has_edge(1, 3),
        gs.has_edge(2, 3)
    );

    gs.local_complement(0);
    println!("Star after LC on center:");
    println!(
        "  edges among leaves (1-2, 1-3, 2-3): {}, {}, {}",
        gs.has_edge(1, 2),
        gs.has_edge(1, 3),
        gs.has_edge(2, 3)
    );
    println!("  total edges: {}", gs.num_edges());

    // Involution: apply again to restore
    gs.local_complement(0);
    println!(
        "After second LC (involution): same as original? {}",
        gs == star
    );

    // -- Vertex deletion -----------------------------------------------------

    println!("\n=== Vertex Deletion ===\n");

    let reduced = linear.delete_vertex(1);
    println!("Linear 0-1-2-3, delete vertex 1:");
    println!(
        "  result has {} qubits, {} edges",
        reduced.num_qubits(),
        reduced.num_edges()
    );

    // -- Entanglement analysis -----------------------------------------------

    println!("\n=== Entanglement Analysis ===\n");

    // Product state
    let product = GraphState::empty(4);
    println!("Product state |+>^4:");
    println!(
        "  Schmidt rank log2 across {{0,1}}|{{2,3}}: {}",
        product.schmidt_rank_log2(&[0, 1])
    );

    // Bell pair
    let mut bell = GraphState::empty(2);
    bell.toggle_edge(0, 1);
    println!("\nBell pair (edge 0-1):");
    println!(
        "  Schmidt rank log2 across {{0}}|{{1}}: {}",
        bell.schmidt_rank_log2(&[0])
    );
    println!("  Entropy (nats): {:.4}", bell.entanglement_entropy(&[0]));

    // Ring bipartitions
    println!("\nRing(4) bipartitions:");
    println!(
        "  {{0,1}}|{{2,3}}: rank = {}",
        ring.schmidt_rank_log2(&[0, 1])
    );
    println!("  {{0}}|{{1,2,3}}: rank = {}", ring.schmidt_rank_log2(&[0]));

    // Linear cluster bipartitions
    println!("\nLinear(4) bipartitions:");
    println!(
        "  {{0,1}}|{{2,3}}: rank = {}",
        linear.schmidt_rank_log2(&[0, 1])
    );
    println!(
        "  {{0}}|{{1,2,3}}: rank = {}",
        linear.schmidt_rank_log2(&[0])
    );

    // -- ZX conversion -------------------------------------------------------

    println!("\n=== ZX Conversion ===\n");

    let zx = linear.to_zx_graph();
    let stats = graph_stats(&zx);
    println!("Linear cluster -> ZX diagram:");
    println!(
        "  {} vertices, {} edges",
        stats.num_vertices, stats.num_edges
    );

    let roundtrip = GraphState::from_zx_graph(&zx).unwrap();
    println!("  roundtrip matches: {}", roundtrip == linear);

    // -- Adjacency matrix display --------------------------------------------

    println!("\n=== Adjacency Matrix (Star) ===\n");
    println!("{star}");
}
