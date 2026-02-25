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

//! Layout algorithms for ZX diagram visualization.

use std::collections::HashMap;

use quizx::graph::{GraphLike, V};

/// Layout algorithm to use for positioning vertices.
#[derive(Debug, Clone, Copy, Default)]
pub enum LayoutAlgorithm {
    /// Use QuiZX's `row()` and `qubit()` coordinates from `to_graph()`.
    /// Best for graphs that were converted from circuits.
    #[default]
    FromGraph,
    /// Simple force-directed layout for graphs that lack circuit coordinates
    /// (e.g., after heavy simplification).
    ForceDirected,
}

/// Options for layout computation.
#[derive(Debug, Clone)]
pub struct LayoutOptions {
    /// Horizontal spacing between time steps (pixels).
    pub x_spacing: f64,
    /// Vertical spacing between qubit wires (pixels).
    pub y_spacing: f64,
    /// Padding around the diagram (pixels).
    pub padding: f64,
    /// Number of iterations for force-directed layout.
    pub force_iterations: usize,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            x_spacing: 80.0,
            y_spacing: 60.0,
            padding: 40.0,
            force_iterations: 100,
        }
    }
}

/// Compute layout positions for all vertices in a ZX graph.
///
/// Returns a map from vertex ID to (x, y) pixel coordinates.
#[must_use]
pub fn compute_layout(
    graph: &impl GraphLike,
    algorithm: LayoutAlgorithm,
    options: &LayoutOptions,
) -> HashMap<V, (f64, f64)> {
    match algorithm {
        LayoutAlgorithm::FromGraph => from_graph_layout(graph, options),
        LayoutAlgorithm::ForceDirected => force_directed_layout(graph, options),
    }
}

/// Use QuiZX's built-in coordinates from `to_graph()`.
///
/// The graph stores `row` (time/x) and `qubit` (wire/y) for each vertex
/// when converted from a circuit. We scale these to pixel coordinates.
fn from_graph_layout(graph: &impl GraphLike, options: &LayoutOptions) -> HashMap<V, (f64, f64)> {
    let mut positions = HashMap::new();

    for v in graph.vertices() {
        let row = graph.row(v);
        let qubit = graph.qubit(v);
        let x = options.padding + row * options.x_spacing;
        let y = options.padding + qubit * options.y_spacing;
        positions.insert(v, (x, y));
    }

    positions
}

/// Simple force-directed layout using spring-embedding.
///
/// Uses repulsion (inverse-square between all pairs) and
/// attraction (linear spring along edges) with damping.
fn force_directed_layout(
    graph: &impl GraphLike,
    options: &LayoutOptions,
) -> HashMap<V, (f64, f64)> {
    let vertices: Vec<V> = graph.vertices().collect();
    let n = vertices.len();

    if n == 0 {
        return HashMap::new();
    }

    // Initialize positions in a grid
    let cols = (n as f64).sqrt().ceil() as usize;
    let mut pos: Vec<(f64, f64)> = vertices
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let col = i % cols;
            let row = i / cols;
            (
                options.padding + col as f64 * options.x_spacing,
                options.padding + row as f64 * options.y_spacing,
            )
        })
        .collect();

    // Build vertex index lookup
    let v_to_idx: HashMap<V, usize> = vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();

    // Collect edges
    let edges: Vec<(usize, usize)> = graph
        .edges()
        .filter_map(|(s, t, _)| {
            let si = v_to_idx.get(&s)?;
            let ti = v_to_idx.get(&t)?;
            Some((*si, *ti))
        })
        .collect();

    let repulsion = 5000.0;
    let attraction = 0.01;
    let damping = 0.9;
    let min_dist = 1.0;

    for _ in 0..options.force_iterations {
        let mut forces = vec![(0.0_f64, 0.0_f64); n];

        // Repulsion between all pairs
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = pos[i].0 - pos[j].0;
                let dy = pos[i].1 - pos[j].1;
                let dist = (dx * dx + dy * dy).sqrt().max(min_dist);
                let force = repulsion / (dist * dist);
                let fx = force * dx / dist;
                let fy = force * dy / dist;
                forces[i].0 += fx;
                forces[i].1 += fy;
                forces[j].0 -= fx;
                forces[j].1 -= fy;
            }
        }

        // Attraction along edges
        for &(i, j) in &edges {
            let dx = pos[j].0 - pos[i].0;
            let dy = pos[j].1 - pos[i].1;
            let dist = (dx * dx + dy * dy).sqrt().max(min_dist);
            let force = attraction * dist;
            let fx = force * dx / dist;
            let fy = force * dy / dist;
            forces[i].0 += fx;
            forces[i].1 += fy;
            forces[j].0 -= fx;
            forces[j].1 -= fy;
        }

        // Apply forces with damping
        for i in 0..n {
            pos[i].0 += forces[i].0 * damping;
            pos[i].1 += forces[i].1 * damping;
        }
    }

    // Normalize so minimum position is at padding
    let min_x = pos.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let min_y = pos.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);

    vertices
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            (
                v,
                (
                    pos[i].0 - min_x + options.padding,
                    pos[i].1 - min_y + options.padding,
                ),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::from_adjacency_matrix;

    #[test]
    fn test_from_graph_layout() {
        #[rustfmt::skip]
        let adj = vec![
            false, true,
            true,  false,
        ];
        let g = from_adjacency_matrix(&adj, 2);
        let layout = compute_layout(&g, LayoutAlgorithm::FromGraph, &LayoutOptions::default());
        assert_eq!(layout.len(), g.num_vertices());
    }

    #[test]
    fn test_force_directed_layout() {
        #[rustfmt::skip]
        let adj = vec![
            false, true,
            true,  false,
        ];
        let g = from_adjacency_matrix(&adj, 2);
        let layout = compute_layout(
            &g,
            LayoutAlgorithm::ForceDirected,
            &LayoutOptions::default(),
        );
        assert_eq!(layout.len(), g.num_vertices());
    }
}
