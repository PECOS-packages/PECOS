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

//! Cup, cap, and snake equation (teleportation) diagrams using curved edges.
//!
//! Demonstrates:
//! - Cup (Bell state): two outputs joined by a downward curve through a Z spider
//! - Cap (Bell effect): two inputs joined by an upward curve through a Z spider
//! - Snake equation: cap composed with cup yields the identity (teleportation)

use std::collections::HashMap;

use num_traits::Zero;
use pecos_zx::simplify;
use pecos_zx::viz::Renderer;
use pecos_zx::{GraphLike, VType, ZxGraph};

/// Build a cup diagram: a Z spider with two output wires, curving outward.
///
/// ```text
///          ⌒--- out[0]
///         /
///    Z --+
///         \
///          ⌣--- out[1]
/// ```
fn build_cup() -> (ZxGraph, HashMap<(usize, usize), f64>) {
    let mut g = ZxGraph::new();

    // Z spider on the left (source of the cup)
    let z = g.add_vertex(VType::Z);
    g.set_coord(z, (0.0, 0.5));

    // Two boundary outputs on the right
    let out0 = g.add_vertex(VType::B);
    g.set_coord(out0, (1.0, 0.0));
    let out1 = g.add_vertex(VType::B);
    g.set_coord(out1, (1.0, 1.0));

    g.add_edge(z, out0);
    g.add_edge(z, out1);
    g.set_outputs(vec![out0, out1]);

    let mut curved = HashMap::new();
    curved.insert((z, out0), -40.0); // bow outward (upward for top arm)
    curved.insert((z, out1), 40.0); // bow outward (downward for bottom arm)

    (g, curved)
}

/// Build a cap diagram: a Z spider with two input wires, curved upward.
///
/// ```text
///        Z(0)
///      /       \
///     |         |
///   in[0]    in[1]
/// ```
fn build_cap() -> (ZxGraph, HashMap<(usize, usize), f64>) {
    let mut g = ZxGraph::new();

    // Two boundary inputs at left
    let in0 = g.add_vertex(VType::B);
    g.set_coord(in0, (0.0, 0.0));
    let in1 = g.add_vertex(VType::B);
    g.set_coord(in1, (0.0, 1.0));

    // Z spider at center right
    let z = g.add_vertex(VType::Z);
    g.set_coord(z, (1.0, 0.5));

    g.add_edge(in0, z);
    g.add_edge(in1, z);
    g.set_inputs(vec![in0, in1]);

    // Curved edges to form the cap shape
    let mut curved = HashMap::new();
    curved.insert((in0, z), -40.0); // bow upward
    curved.insert((in1, z), 40.0); // bow downward

    (g, curved)
}

/// Build the snake equation diagram: an S-shaped zigzag from top-left to bottom-right.
///
/// ```text
///   in[0] ── cap ──╲
///                    ╲  (S-curve via mid)
///                    ╱
///              cup ─╱── out[0]
/// ```
///
/// The snake equation shows that composing a cap with a cup yields the identity:
/// the wire enters on qubit 0 (top), bends down to qubit 1 (bottom) via the cap,
/// and exits on qubit 1 as the output. The S-shaped path IS the identity.
fn build_snake() -> (ZxGraph, HashMap<(usize, usize), f64>) {
    let mut g = ZxGraph::new();

    // Input boundary at top-left (qubit 0)
    let inp = g.add_vertex(VType::B);
    g.set_coord(inp, (0.0, 0.0));

    // Cap spider on the top wire (qubit 0)
    let cap = g.add_vertex(VType::Z);
    g.set_coord(cap, (1.0, 0.0));

    // Mid spider at the inflection point (between qubits)
    let mid = g.add_vertex(VType::Z);
    g.set_coord(mid, (2.0, 0.5));

    // Cup spider on the bottom wire (qubit 1)
    let cup = g.add_vertex(VType::Z);
    g.set_coord(cup, (3.0, 1.0));

    // Output boundary at bottom-right (qubit 1)
    let out = g.add_vertex(VType::B);
    g.set_coord(out, (4.0, 1.0));

    // Straight edges on the horizontal wires
    g.add_edge(inp, cap);
    g.add_edge(cup, out);

    // S-curve: cap -> mid -> cup
    g.add_edge(cap, mid);
    g.add_edge(mid, cup);

    g.set_inputs(vec![inp]);
    g.set_outputs(vec![out]);

    let mut curved = HashMap::new();
    // Upper arc: bows right-and-up to form the top of the S
    curved.insert((cap, mid), -40.0);
    // Lower arc: bows left-and-down to form the bottom of the S
    curved.insert((mid, cup), 40.0);

    (g, curved)
}

fn main() {
    println!("=== Teleportation: Cups, Caps, and the Snake Equation ===\n");

    let mut r = Renderer::default();
    r.set_output_dir("exp/pecos-zx/examples/output");

    // Cup (Bell state)
    let (cup_graph, cup_curves) = build_cup();
    r.svg.curved_edges = cup_curves;
    r.render(&cup_graph, "teleportation_cup");

    // Cap (Bell effect)
    let (cap_graph, cap_curves) = build_cap();
    r.svg.curved_edges = cap_curves;
    r.render(&cap_graph, "teleportation_cap");

    // Snake equation (before simplification)
    let (mut snake_graph, snake_curves) = build_snake();
    r.svg.curved_edges = snake_curves.clone();
    r.render(&snake_graph, "teleportation_snake");

    // Snake -- identity spiders hidden via render option
    r.svg.hide_identities = true;
    r.ascii.hide_identities = true;
    r.render(&snake_graph, "teleportation_snake_bent");

    // Snake -- identity spiders removed by changing vertex types (manual approach)
    let mut snake_manual = snake_graph.clone();
    for v in snake_manual.vertices().collect::<Vec<_>>() {
        if matches!(snake_manual.vertex_type(v), VType::Z | VType::X)
            && snake_manual.degree(v) == 2
            && snake_manual.phase(v).is_zero()
        {
            snake_manual.set_vertex_type(v, VType::B);
        }
    }
    r.svg.hide_identities = false;
    r.ascii.hide_identities = false;
    r.svg.curved_edges = snake_curves;
    r.svg.show_boundary_labels = false;
    r.render(&snake_manual, "teleportation_snake_manual");

    // Snake -- fully simplified (should reduce to a single straight wire)
    simplify::clifford_simp(&mut snake_graph);
    r.svg.show_boundary_labels = true;
    r.svg.curved_edges = HashMap::new();
    r.render(&snake_graph, "teleportation_snake_simplified");
}
