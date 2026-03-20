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

//! Demonstrates individual ZX rewrite rules with before/after output.
//!
//! Shows:
//! - Spider fusion (merging adjacent same-colored spiders)
//! - Identity removal

use pecos_zx::basic_rules;
use pecos_zx::graph::graph_stats;
use pecos_zx::viz::Renderer;
use pecos_zx::{GraphLike, VType, ZxGraph};

fn main() {
    println!("=== ZX Rewrite Rules ===\n");

    demo_spider_fusion();
    demo_identity_removal();
}

/// Demonstrate spider fusion: two adjacent Z-spiders merge into one.
fn demo_spider_fusion() {
    // Create: B -- Z(0) -- Z(pi/2) -- B
    let mut g = ZxGraph::new();
    let b0 = g.add_vertex(VType::B);
    g.set_coord(b0, (0.0, 0.0));
    let z0 = g.add_vertex(VType::Z);
    g.set_coord(z0, (1.0, 0.0));
    let z1 = g.add_vertex(VType::Z);
    g.set_coord(z1, (2.0, 0.0));
    g.set_phase(z1, (1, 2)); // pi/2
    let b1 = g.add_vertex(VType::B);
    g.set_coord(b1, (3.0, 0.0));

    g.add_edge(b0, z0);
    g.add_edge(z0, z1);
    g.add_edge(z1, b1);
    g.set_inputs(vec![b0]);
    g.set_outputs(vec![b1]);

    let stats_before = graph_stats(&g);
    println!(
        "Spider fusion before: {} vertices, {} edges",
        stats_before.num_vertices, stats_before.num_edges
    );

    let mut r = Renderer::default();
    r.set_output_dir("exp/pecos-zx/examples/output");
    r.render(&g, "rewrite_fusion_before");

    // Apply spider fusion
    let fused = basic_rules::spider_fusion(&mut g, z0, z1);
    println!("\nApplied spider_fusion: {fused}");

    let stats_after = graph_stats(&g);
    println!(
        "Spider fusion after: {} vertices, {} edges\n",
        stats_after.num_vertices, stats_after.num_edges
    );

    r.render(&g, "rewrite_fusion_after");
}

/// Demonstrate identity removal: a phase-zero spider with degree 2 is removed.
fn demo_identity_removal() {
    // Create: B -- Z(0) -- B  (Z spider with zero phase and degree 2)
    let mut g = ZxGraph::new();
    let b0 = g.add_vertex(VType::B);
    g.set_coord(b0, (0.0, 0.0));
    let z0 = g.add_vertex(VType::Z);
    g.set_coord(z0, (1.0, 0.0));
    // phase defaults to 0
    let b1 = g.add_vertex(VType::B);
    g.set_coord(b1, (2.0, 0.0));

    g.add_edge(b0, z0);
    g.add_edge(z0, b1);
    g.set_inputs(vec![b0]);
    g.set_outputs(vec![b1]);

    let stats_before = graph_stats(&g);
    println!(
        "\nIdentity removal before: {} vertices, {} edges",
        stats_before.num_vertices, stats_before.num_edges
    );

    let mut r = Renderer::default();
    r.set_output_dir("exp/pecos-zx/examples/output");
    r.render(&g, "rewrite_id_before");

    // Apply identity removal
    let removed = basic_rules::remove_id(&mut g, z0);
    println!("\nApplied remove_id: {removed}");

    let stats_after = graph_stats(&g);
    println!(
        "Identity removal after: {} vertices, {} edges\n",
        stats_after.num_vertices, stats_after.num_edges
    );

    r.render(&g, "rewrite_id_after");
}
