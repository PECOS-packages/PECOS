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

//! Noise model for annotating ZX graph edges with error probabilities.
//!
//! Provides [`ErrorRates`] for per-edge Pauli error probabilities and
//! [`NoiseModel`] for collecting them across a graph.

use std::collections::HashMap;

use quizx::graph::{GraphLike, VType};

use crate::ZxGraph;

/// Per-edge Pauli error probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorRates {
    /// Probability of an X error on this edge.
    pub px: f64,
    /// Probability of a Y error on this edge.
    pub py: f64,
    /// Probability of a Z error on this edge.
    pub pz: f64,
}

impl ErrorRates {
    /// Create error rates with explicit probabilities.
    #[must_use]
    pub fn new(px: f64, py: f64, pz: f64) -> Self {
        Self { px, py, pz }
    }

    /// Create depolarizing error rates where each Pauli gets `p / 3`.
    #[must_use]
    pub fn depolarizing(p: f64) -> Self {
        let p3 = p / 3.0;
        Self {
            px: p3,
            py: p3,
            pz: p3,
        }
    }
}

/// A noise model mapping graph edges to error rates.
///
/// Edge keys are stored in canonical form `(min(u, v), max(u, v))` to match
/// the convention used by [`PauliWeb::set_edge`](quizx::detection_webs::PauliWeb::set_edge).
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// Error rates keyed by canonical edge `(min, max)`.
    pub edge_errors: HashMap<(usize, usize), ErrorRates>,
}

impl NoiseModel {
    /// Create an empty noise model.
    #[must_use]
    pub fn new() -> Self {
        Self {
            edge_errors: HashMap::new(),
        }
    }

    /// Set the error rates for an edge, canonicalizing the key.
    pub fn set_edge(&mut self, u: usize, v: usize, rates: ErrorRates) {
        let key = (u.min(v), u.max(v));
        self.edge_errors.insert(key, rates);
    }

    /// Look up the error rates for an edge.
    #[must_use]
    pub fn edge(&self, u: usize, v: usize) -> Option<&ErrorRates> {
        let key = (u.min(v), u.max(v));
        self.edge_errors.get(&key)
    }

    /// Build a noise model with uniform depolarizing noise on all internal edges.
    ///
    /// Internal edges are those where neither endpoint is a boundary vertex
    /// (`VType::B`). Boundary edges represent I/O wires, not physical gates.
    #[must_use]
    pub fn uniform_depolarizing(graph: &ZxGraph, p: f64) -> Self {
        let rates = ErrorRates::depolarizing(p);
        let mut model = Self::new();

        for (s, t, _ety) in graph.edges() {
            if graph.vertex_type(s) != VType::B && graph.vertex_type(t) != VType::B {
                model
                    .edge_errors
                    .insert((s.min(t), s.max(t)), rates.clone());
            }
        }

        model
    }
}

impl Default for NoiseModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quizx::graph::GraphLike;

    #[test]
    fn test_depolarizing_rates() {
        let rates = ErrorRates::depolarizing(0.03);
        let expected = 0.01;
        assert!((rates.px - expected).abs() < 1e-12);
        assert!((rates.py - expected).abs() < 1e-12);
        assert!((rates.pz - expected).abs() < 1e-12);
    }

    #[test]
    fn test_edge_canonical_order() {
        let mut model = NoiseModel::new();
        model.set_edge(5, 3, ErrorRates::new(0.1, 0.2, 0.3));

        // Should be accessible in both orders
        assert!(model.edge(3, 5).is_some());
        assert!(model.edge(5, 3).is_some());
        assert_eq!(model.edge(3, 5).unwrap().px, 0.1);
    }

    #[test]
    fn test_set_edge_overwrites() {
        let mut model = NoiseModel::new();
        model.set_edge(1, 2, ErrorRates::new(0.1, 0.2, 0.3));
        model.set_edge(1, 2, ErrorRates::new(0.4, 0.5, 0.6));

        let rates = model.edge(1, 2).unwrap();
        assert_eq!(rates.px, 0.4);
        assert_eq!(rates.py, 0.5);
        assert_eq!(rates.pz, 0.6);
    }

    #[test]
    fn test_missing_edge_returns_none() {
        let model = NoiseModel::new();
        assert!(model.edge(0, 1).is_none());
    }

    #[test]
    fn test_uniform_skips_boundaries() {
        // Build a minimal graph with boundary and internal vertices
        let mut g = ZxGraph::new();
        let b0 = g.add_vertex(VType::B); // boundary
        let z0 = g.add_vertex(VType::Z); // internal
        let z1 = g.add_vertex(VType::Z); // internal
        let b1 = g.add_vertex(VType::B); // boundary

        g.add_edge(b0, z0);
        g.add_edge(z0, z1);
        g.add_edge(z1, b1);

        let model = NoiseModel::uniform_depolarizing(&g, 0.03);

        // Only the internal edge (z0, z1) should have noise
        assert!(model.edge(z0, z1).is_some());
        // Boundary edges should be skipped
        assert!(model.edge(b0, z0).is_none());
        assert!(model.edge(z1, b1).is_none());
    }
}
