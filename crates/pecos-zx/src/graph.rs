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

//! ZX graph helpers and metadata.

use std::collections::HashMap;

use quizx::graph::{GraphLike, V, VType};

use crate::ZxGraph;

/// PECOS-specific metadata for ZX graph vertices.
///
/// This lives alongside the QuiZX `Graph` rather than wrapping it,
/// allowing direct use of `GraphLike` methods while storing additional
/// information like layout positions and annotations.
#[derive(Debug, Clone, Default)]
pub struct ZxMetadata {
    /// Layout positions for vertices (vertex_id -> (x, y)).
    pub positions: HashMap<usize, (f64, f64)>,
    /// Labels for boundary vertices (vertex_id -> label string).
    pub boundary_labels: HashMap<usize, String>,
    /// Pauli web annotations for edges.
    pub web_annotations: HashMap<(usize, usize), String>,
}

/// Information about a single spider in a ZX graph.
#[derive(Debug, Clone)]
pub struct SpiderInfo {
    pub vertex: V,
    pub vertex_type: VType,
    pub phase: String,
    pub degree: usize,
}

/// Summary statistics for a ZX graph.
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub num_vertices: usize,
    pub num_edges: usize,
    pub num_z_spiders: usize,
    pub num_x_spiders: usize,
    pub num_h_boxes: usize,
    pub num_boundaries: usize,
    pub num_inputs: usize,
    pub num_outputs: usize,
}

/// Create a graph state ZX diagram from an adjacency matrix.
///
/// Each qubit becomes a Z spider with a boundary input and output.
/// Edges between qubits in the adjacency matrix become normal edges
/// between the corresponding Z spiders.
///
/// The adjacency matrix is given as a flat `n x n` slice in row-major order.
#[must_use]
pub fn from_adjacency_matrix(adj: &[bool], n: usize) -> ZxGraph {
    assert_eq!(adj.len(), n * n, "adjacency matrix must be n x n");

    let mut g = ZxGraph::new();
    let mut inputs = Vec::with_capacity(n);
    let mut outputs = Vec::with_capacity(n);
    let mut spiders = Vec::with_capacity(n);

    // Create boundary + spider for each qubit
    for i in 0..n {
        let inp = g.add_vertex(VType::B);
        g.set_coord(inp, (0.0, i as f64));
        inputs.push(inp);

        let spider = g.add_vertex(VType::Z);
        g.set_coord(spider, (1.0, i as f64));
        spiders.push(spider);

        let out = g.add_vertex(VType::B);
        g.set_coord(out, (2.0, i as f64));
        outputs.push(out);

        g.add_edge(inp, spider);
        g.add_edge(spider, out);
    }

    // Add edges from adjacency matrix (upper triangle only to avoid duplicates)
    for i in 0..n {
        for j in (i + 1)..n {
            if adj[i * n + j] {
                g.add_edge(spiders[i], spiders[j]);
            }
        }
    }

    g.set_inputs(inputs);
    g.set_outputs(outputs);
    g
}

/// Get information about a specific spider.
#[must_use]
pub fn spider_info(g: &impl GraphLike, v: V) -> SpiderInfo {
    SpiderInfo {
        vertex: v,
        vertex_type: g.vertex_type(v),
        phase: format!("{}", g.phase(v)),
        degree: g.degree(v),
    }
}

/// Compute summary statistics for a ZX graph.
#[must_use]
pub fn graph_stats(g: &impl GraphLike) -> GraphStats {
    let mut stats = GraphStats {
        num_vertices: g.num_vertices(),
        num_edges: g.num_edges(),
        num_z_spiders: 0,
        num_x_spiders: 0,
        num_h_boxes: 0,
        num_boundaries: 0,
        num_inputs: g.inputs().len(),
        num_outputs: g.outputs().len(),
    };

    for v in g.vertices() {
        match g.vertex_type(v) {
            VType::Z => stats.num_z_spiders += 1,
            VType::X => stats.num_x_spiders += 1,
            VType::H => stats.num_h_boxes += 1,
            VType::B => stats.num_boundaries += 1,
            _ => {}
        }
    }

    stats
}

/// Count the total number of spiders (Z + X) in a ZX graph.
#[must_use]
pub fn num_spiders(g: &impl GraphLike) -> usize {
    g.vertices()
        .filter(|&v| matches!(g.vertex_type(v), VType::Z | VType::X))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_adjacency_matrix_linear() {
        // Linear cluster: 0-1-2
        #[rustfmt::skip]
        let adj = vec![
            false, true,  false,
            true,  false, true,
            false, true,  false,
        ];
        let g = from_adjacency_matrix(&adj, 3);

        assert_eq!(g.inputs().len(), 3);
        assert_eq!(g.outputs().len(), 3);
        // 3 inputs + 3 spiders + 3 outputs = 9 vertices
        assert_eq!(g.num_vertices(), 9);
        // 3 input-spider + 3 spider-output + 2 spider-spider = 8 edges
        assert_eq!(g.num_edges(), 8);
    }

    #[test]
    fn test_from_adjacency_matrix_star() {
        // Star: center=0 connected to 1,2,3
        #[rustfmt::skip]
        let adj = vec![
            false, true,  true,  true,
            true,  false, false, false,
            true,  false, false, false,
            true,  false, false, false,
        ];
        let g = from_adjacency_matrix(&adj, 4);
        assert_eq!(g.num_vertices(), 12);
        // 4 input-spider + 4 spider-output + 3 spider-spider = 11
        assert_eq!(g.num_edges(), 11);
    }

    #[test]
    fn test_graph_stats() {
        #[rustfmt::skip]
        let adj = vec![
            false, true,
            true,  false,
        ];
        let g = from_adjacency_matrix(&adj, 2);
        let stats = graph_stats(&g);

        assert_eq!(stats.num_z_spiders, 2);
        assert_eq!(stats.num_boundaries, 4);
        assert_eq!(stats.num_inputs, 2);
        assert_eq!(stats.num_outputs, 2);
    }

    #[test]
    fn test_num_spiders() {
        let adj = vec![false; 9];
        let g = from_adjacency_matrix(&adj, 3);
        assert_eq!(num_spiders(&g), 3);
    }
}
