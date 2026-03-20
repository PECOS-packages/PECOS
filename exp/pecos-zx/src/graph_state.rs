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

//! Graph state representation and analysis.
//!
//! A [`GraphState`] represents a stabilizer state that can be described by a
//! simple undirected graph: start with every qubit in |+>, then apply CZ gates
//! for each edge. The adjacency matrix (over GF(2)) fully characterizes the
//! state.
//!
//! This module provides:
//! - Construction from adjacency matrices
//! - Graph manipulation (edge toggling, local complementation, vertex deletion)
//! - Entanglement analysis via GF(2) rank of biadjacency submatrices
//! - Conversion to/from ZX diagrams

use std::collections::BTreeSet;
use std::fmt;

use num_traits::Zero;
use quizx::graph::{EType, GraphLike, VType};
use quizx::linalg::Mat2;

use crate::ZxGraph;

/// A graph state on `n` qubits, represented by its `n x n` symmetric adjacency
/// matrix over GF(2) with zero diagonal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphState {
    adj: Mat2,
}

/// Errors arising from graph state operations.
#[derive(Debug, thiserror::Error)]
pub enum GraphStateError {
    /// The ZX diagram does not have graph-state structure.
    #[error("not a graph state: {0}")]
    NotGraphState(String),
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl GraphState {
    /// Create a graph state with `n` qubits and no edges (product state |+>^n).
    #[must_use]
    pub fn empty(n: usize) -> Self {
        Self {
            adj: Mat2::zeros(n, n),
        }
    }

    /// Create a graph state from a `Mat2` adjacency matrix.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square, not symmetric, or has nonzero diagonal.
    #[must_use]
    pub fn from_mat2(adj: Mat2) -> Self {
        let n = adj.num_rows();
        assert_eq!(
            adj.num_cols(),
            n,
            "adjacency matrix must be square, got {n} x {}",
            adj.num_cols()
        );
        for i in 0..n {
            assert_eq!(adj[(i, i)], 0, "diagonal entry ({i}, {i}) must be zero");
            for j in (i + 1)..n {
                assert_eq!(
                    adj[(i, j)],
                    adj[(j, i)],
                    "matrix must be symmetric at ({i}, {j})"
                );
            }
        }
        Self { adj }
    }

    /// Create a graph state from a flat row-major `n x n` boolean slice.
    ///
    /// Only the upper triangle is read; the lower triangle is set to match,
    /// enforcing symmetry. Diagonal entries are ignored (forced to zero).
    ///
    /// # Panics
    ///
    /// Panics if `adj.len() != n * n`.
    #[must_use]
    pub fn from_flat(adj: &[bool], n: usize) -> Self {
        assert_eq!(adj.len(), n * n, "flat adjacency must have n*n entries");
        let mat = Mat2::build(n, n, |i, j| {
            if i == j {
                false
            } else if i < j {
                adj[i * n + j]
            } else {
                adj[j * n + i]
            }
        });
        Self { adj: mat }
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

impl GraphState {
    /// Number of qubits.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.adj.num_rows()
    }

    /// Read-only access to the adjacency matrix.
    #[must_use]
    pub fn adjacency(&self) -> &Mat2 {
        &self.adj
    }

    /// Whether qubits `i` and `j` are connected.
    #[must_use]
    pub fn has_edge(&self, i: usize, j: usize) -> bool {
        self.adj[(i, j)] != 0
    }

    /// List of neighbors of qubit `v`.
    #[must_use]
    pub fn neighbors(&self, v: usize) -> Vec<usize> {
        (0..self.num_qubits())
            .filter(|&u| self.adj[(v, u)] != 0)
            .collect()
    }

    /// Degree of qubit `v` (number of neighbors).
    #[must_use]
    pub fn degree(&self, v: usize) -> usize {
        (0..self.num_qubits())
            .filter(|&u| self.adj[(v, u)] != 0)
            .count()
    }

    /// Total number of edges.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        let n = self.num_qubits();
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.adj[(i, j)] != 0 {
                    count += 1;
                }
            }
        }
        count
    }

    /// Export the adjacency matrix as a flat row-major boolean vector.
    #[must_use]
    pub fn to_flat(&self) -> Vec<bool> {
        let n = self.num_qubits();
        let mut out = vec![false; n * n];
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = self.adj[(i, j)] != 0;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Manipulation
// ---------------------------------------------------------------------------

impl GraphState {
    /// Toggle edge between qubits `i` and `j`. No-op if `i == j`.
    pub fn toggle_edge(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        self.adj[(i, j)] ^= 1;
        self.adj[(j, i)] ^= 1;
    }

    /// Apply local complementation on vertex `v`.
    ///
    /// For every pair of distinct neighbors `(i, j)` of `v`, the edge `(i, j)`
    /// is toggled. This corresponds to the graph-theoretic operation associated
    /// with applying a local Clifford gate on qubit `v`.
    ///
    /// Applying local complementation twice on the same vertex restores the
    /// original graph.
    pub fn local_complement(&mut self, v: usize) {
        let nbrs = self.neighbors(v);
        for (idx, &i) in nbrs.iter().enumerate() {
            for &j in &nbrs[idx + 1..] {
                self.toggle_edge(i, j);
            }
        }
    }

    /// Return a new graph state with vertex `v` removed and remaining vertices
    /// reindexed.
    #[must_use]
    pub fn delete_vertex(&self, v: usize) -> Self {
        let n = self.num_qubits();
        assert!(v < n, "vertex {v} out of range for {n}-qubit graph state");
        let mat = Mat2::build(n - 1, n - 1, |i, j| {
            let si = if i >= v { i + 1 } else { i };
            let sj = if j >= v { j + 1 } else { j };
            self.adj[(si, sj)] != 0
        });
        Self { adj: mat }
    }
}

// ---------------------------------------------------------------------------
// Entanglement analysis
// ---------------------------------------------------------------------------

impl GraphState {
    /// Compute log2 of the Schmidt rank across the bipartition `A | B`, where
    /// `partition_a` lists the qubit indices in A and B is the complement.
    ///
    /// This equals the GF(2) rank of the biadjacency submatrix `adj[A, B]`,
    /// which gives the entanglement in ebits.
    #[must_use]
    pub fn schmidt_rank_log2(&self, partition_a: &[usize]) -> usize {
        let n = self.num_qubits();
        let set_a: BTreeSet<usize> = partition_a.iter().copied().collect();
        let set_b: Vec<usize> = (0..n).filter(|v| !set_a.contains(v)).collect();

        if set_a.is_empty() || set_b.is_empty() {
            return 0;
        }

        let rows: Vec<usize> = set_a.into_iter().collect();
        let biadj = Mat2::build(rows.len(), set_b.len(), |i, j| {
            self.adj[(rows[i], set_b[j])] != 0
        });
        biadj.rank()
    }

    /// Entanglement entropy (in nats) across the bipartition `A | B`.
    ///
    /// For a graph state the entanglement entropy equals
    /// `schmidt_rank_log2(A) * ln(2)`.
    #[must_use]
    pub fn entanglement_entropy(&self, partition_a: &[usize]) -> f64 {
        self.schmidt_rank_log2(partition_a) as f64 * std::f64::consts::LN_2
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

impl GraphState {
    /// Convert to a ZX diagram in graph-state form (Z spiders with boundary
    /// inputs and outputs).
    #[must_use]
    pub fn to_zx_graph(&self) -> ZxGraph {
        crate::graph::from_adjacency_matrix(&self.to_flat(), self.num_qubits())
    }

    /// Extract a `GraphState` from a ZX diagram that has graph-state structure.
    ///
    /// The diagram must consist of:
    /// - Boundary input vertices, each connected to exactly one Z spider
    /// - Z spiders with zero phase, connected among themselves via normal edges
    /// - Boundary output vertices, each connected to exactly one Z spider
    ///
    /// # Errors
    ///
    /// Returns `GraphStateError::NotGraphState` if the diagram does not
    /// conform to graph-state structure.
    pub fn from_zx_graph(g: &ZxGraph) -> Result<Self, GraphStateError> {
        let n = g.inputs().len();
        if n != g.outputs().len() {
            return Err(GraphStateError::NotGraphState(format!(
                "input count ({}) != output count ({})",
                n,
                g.outputs().len()
            )));
        }

        // Map each qubit index to its Z spider vertex
        let mut qubit_to_spider = Vec::with_capacity(n);
        for (qi, &inp) in g.inputs().iter().enumerate() {
            if g.vertex_type(inp) != VType::B {
                return Err(GraphStateError::NotGraphState(format!(
                    "input {qi} is not a boundary vertex"
                )));
            }
            let nbrs: Vec<_> = g.neighbors(inp).collect();
            if nbrs.len() != 1 {
                return Err(GraphStateError::NotGraphState(format!(
                    "input {qi} has {} neighbors, expected 1",
                    nbrs.len()
                )));
            }
            let spider = nbrs[0];
            if g.vertex_type(spider) != VType::Z {
                return Err(GraphStateError::NotGraphState(format!(
                    "input {qi} neighbor is {:?}, expected Z spider",
                    g.vertex_type(spider)
                )));
            }
            if !g.phase(spider).is_zero() {
                return Err(GraphStateError::NotGraphState(format!(
                    "spider for qubit {qi} has non-zero phase {}",
                    g.phase(spider)
                )));
            }
            qubit_to_spider.push(spider);
        }

        // Verify outputs connect to the same spiders
        for (qi, &out) in g.outputs().iter().enumerate() {
            if g.vertex_type(out) != VType::B {
                return Err(GraphStateError::NotGraphState(format!(
                    "output {qi} is not a boundary vertex"
                )));
            }
            let nbrs: Vec<_> = g.neighbors(out).collect();
            if nbrs.len() != 1 {
                return Err(GraphStateError::NotGraphState(format!(
                    "output {qi} has {} neighbors, expected 1",
                    nbrs.len()
                )));
            }
            if nbrs[0] != qubit_to_spider[qi] {
                return Err(GraphStateError::NotGraphState(format!(
                    "output {qi} does not connect to same spider as input {qi}"
                )));
            }
        }

        // Build adjacency from spider-spider edges
        let adj = Mat2::build(n, n, |i, j| {
            if i == j {
                return false;
            }
            match g.edge_type_opt(qubit_to_spider[i], qubit_to_spider[j]) {
                Some(EType::N) => true,
                Some(et) => {
                    // Non-normal edges aren't graph-state structure, but we
                    // treat them as connected for robustness in extraction.
                    // A stricter version could error here.
                    panic!("unexpected edge type {et:?} between spiders {i} and {j}");
                }
                None => false,
            }
        });

        Ok(Self { adj })
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for GraphState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.adj)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Star graph: vertex 0 connected to all others.
    fn star_graph(n: usize) -> GraphState {
        let mut gs = GraphState::empty(n);
        for i in 1..n {
            gs.toggle_edge(0, i);
        }
        gs
    }

    /// Triangle: 0-1, 1-2, 0-2.
    fn triangle_graph() -> GraphState {
        let mut gs = GraphState::empty(3);
        gs.toggle_edge(0, 1);
        gs.toggle_edge(1, 2);
        gs.toggle_edge(0, 2);
        gs
    }

    /// Linear cluster: 0-1-2-..-(n-1).
    fn linear_cluster(n: usize) -> GraphState {
        let mut gs = GraphState::empty(n);
        for i in 0..n.saturating_sub(1) {
            gs.toggle_edge(i, i + 1);
        }
        gs
    }

    /// Ring: 0-1-2-..-(n-1)-0.
    fn ring_graph(n: usize) -> GraphState {
        let mut gs = linear_cluster(n);
        if n >= 2 {
            gs.toggle_edge(0, n - 1);
        }
        gs
    }

    // -----------------------------------------------------------------------
    // Constructor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty() {
        let gs = GraphState::empty(4);
        assert_eq!(gs.num_qubits(), 4);
        assert_eq!(gs.num_edges(), 0);
    }

    #[test]
    fn test_empty_zero_qubits() {
        let gs = GraphState::empty(0);
        assert_eq!(gs.num_qubits(), 0);
        assert_eq!(gs.num_edges(), 0);
        assert!(gs.to_flat().is_empty());
    }

    #[test]
    fn test_empty_one_qubit() {
        let gs = GraphState::empty(1);
        assert_eq!(gs.num_qubits(), 1);
        assert_eq!(gs.num_edges(), 0);
        assert!(gs.neighbors(0).is_empty());
        assert_eq!(gs.degree(0), 0);
    }

    #[test]
    fn test_from_flat_roundtrip() {
        #[rustfmt::skip]
        let flat = vec![
            false, true,  false,
            true,  false, true,
            false, true,  false,
        ];
        let gs = GraphState::from_flat(&flat, 3);
        assert_eq!(gs.to_flat(), flat);
    }

    #[test]
    fn test_from_flat_enforces_symmetry() {
        // Lower triangle has different values -- only upper is read
        #[rustfmt::skip]
        let flat = vec![
            false, true,
            false, false,  // lower triangle says no edge
        ];
        let gs = GraphState::from_flat(&flat, 2);
        assert!(gs.has_edge(0, 1));
        assert!(gs.has_edge(1, 0)); // enforced symmetric
    }

    #[test]
    fn test_from_mat2() {
        let mat = Mat2::build(3, 3, |i, j| (i == 0 && j == 1) || (i == 1 && j == 0));
        let gs = GraphState::from_mat2(mat);
        assert_eq!(gs.num_edges(), 1);
        assert!(gs.has_edge(0, 1));
    }

    #[test]
    #[should_panic(expected = "symmetric")]
    fn test_from_mat2_rejects_asymmetric() {
        let mut mat = Mat2::zeros(2, 2);
        mat[(0, 1)] = 1;
        // mat[(1, 0)] left as 0 => asymmetric
        let _ = GraphState::from_mat2(mat);
    }

    #[test]
    #[should_panic(expected = "square")]
    fn test_from_mat2_rejects_non_square() {
        let mat = Mat2::zeros(2, 3);
        let _ = GraphState::from_mat2(mat);
    }

    #[test]
    #[should_panic(expected = "diagonal")]
    fn test_from_mat2_rejects_nonzero_diagonal() {
        let mut mat = Mat2::zeros(2, 2);
        mat[(0, 0)] = 1;
        let _ = GraphState::from_mat2(mat);
    }

    // -----------------------------------------------------------------------
    // Accessor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_neighbors_and_degree() {
        let gs = star_graph(4);
        assert_eq!(gs.degree(0), 3);
        assert_eq!(gs.neighbors(0), vec![1, 2, 3]);
        assert_eq!(gs.degree(1), 1);
        assert_eq!(gs.neighbors(1), vec![0]);
    }

    #[test]
    fn test_to_flat_roundtrip() {
        let gs = triangle_graph();
        let flat = gs.to_flat();
        let gs2 = GraphState::from_flat(&flat, 3);
        assert_eq!(gs, gs2);
    }

    #[test]
    fn test_num_edges() {
        assert_eq!(triangle_graph().num_edges(), 3);
        assert_eq!(star_graph(4).num_edges(), 3);
        assert_eq!(linear_cluster(4).num_edges(), 3);
        assert_eq!(ring_graph(4).num_edges(), 4);
        assert_eq!(GraphState::empty(5).num_edges(), 0);
    }

    // -----------------------------------------------------------------------
    // Local complementation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_toggle_edge_self_loop_noop() {
        let mut gs = linear_cluster(3);
        let original = gs.clone();
        gs.toggle_edge(1, 1);
        assert_eq!(gs, original);
    }

    #[test]
    fn test_lc_triangle() {
        let mut gs = triangle_graph();
        gs.local_complement(0);
        // Neighbors of 0 are {1, 2}. Toggle (1,2): was on, now off.
        assert!(gs.has_edge(0, 1));
        assert!(gs.has_edge(0, 2));
        assert!(!gs.has_edge(1, 2));
    }

    #[test]
    fn test_lc_star_on_center() {
        let mut gs = star_graph(4);
        // Neighbors of 0: {1, 2, 3}. Toggle all pairs among them.
        gs.local_complement(0);
        // Original had no edges among 1,2,3 -- now they form a triangle
        assert!(gs.has_edge(1, 2));
        assert!(gs.has_edge(1, 3));
        assert!(gs.has_edge(2, 3));
        // Original star edges still present
        assert!(gs.has_edge(0, 1));
        assert!(gs.has_edge(0, 2));
        assert!(gs.has_edge(0, 3));
    }

    #[test]
    fn test_lc_isolated_vertex() {
        let mut gs = linear_cluster(3);
        let original = gs.clone();
        // Vertex 0 in 0-1-2 has only neighbor {1}.
        // LC on vertex 2 has only neighbor {1} -- single neighbor, no pairs.
        gs.local_complement(2);
        assert_eq!(gs, original);
    }

    #[test]
    fn test_lc_zero_degree_vertex() {
        let mut gs = GraphState::empty(3);
        // Add one edge not involving vertex 2
        gs.toggle_edge(0, 1);
        let original = gs.clone();
        // LC on vertex 2 which has no neighbors at all
        gs.local_complement(2);
        assert_eq!(gs, original);
    }

    #[test]
    fn test_lc_involution() {
        let mut gs = ring_graph(5);
        let original = gs.clone();
        gs.local_complement(2);
        assert_ne!(gs, original); // should differ
        gs.local_complement(2);
        assert_eq!(gs, original); // restored
    }

    // -----------------------------------------------------------------------
    // Vertex deletion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_delete_middle_of_linear() {
        // 0-1-2 => delete 1 => two isolated vertices {0, 2}
        let gs = linear_cluster(3);
        let reduced = gs.delete_vertex(1);
        assert_eq!(reduced.num_qubits(), 2);
        assert_eq!(reduced.num_edges(), 0);
    }

    #[test]
    fn test_delete_leaf() {
        // Star(4): 0-{1,2,3} => delete leaf 3 => star on 0-{1,2}
        let gs = star_graph(4);
        let reduced = gs.delete_vertex(3);
        assert_eq!(reduced.num_qubits(), 3);
        assert_eq!(reduced.num_edges(), 2);
        assert!(reduced.has_edge(0, 1));
        assert!(reduced.has_edge(0, 2));
    }

    #[test]
    fn test_delete_from_triangle() {
        // Triangle 0-1-2 => delete 0 => edge 1-2 remains (reindexed to 0-1)
        let gs = triangle_graph();
        let reduced = gs.delete_vertex(0);
        assert_eq!(reduced.num_qubits(), 2);
        assert_eq!(reduced.num_edges(), 1);
        assert!(reduced.has_edge(0, 1));
    }

    // -----------------------------------------------------------------------
    // Entanglement tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_product_state_rank() {
        let gs = GraphState::empty(4);
        assert_eq!(gs.schmidt_rank_log2(&[0, 1]), 0);
        assert_eq!(gs.schmidt_rank_log2(&[0]), 0);
    }

    #[test]
    fn test_bell_pair_rank() {
        // Single edge 0-1
        let mut gs = GraphState::empty(2);
        gs.toggle_edge(0, 1);
        assert_eq!(gs.schmidt_rank_log2(&[0]), 1);
        assert_eq!(gs.schmidt_rank_log2(&[1]), 1);
    }

    #[test]
    fn test_star_rank() {
        let gs = star_graph(4);
        // Cut {0} | {1,2,3}: biadjacency is [1 1 1], rank 1
        assert_eq!(gs.schmidt_rank_log2(&[0]), 1);
        // Cut {1,2} | {0,3}: biadjacency has rows for 1->{0,3} and 2->{0,3}
        // Row 1: [1, 0], Row 2: [1, 0] => rank 1
        assert_eq!(gs.schmidt_rank_log2(&[1, 2]), 1);
    }

    #[test]
    fn test_ring4_rank() {
        // Ring 0-1-2-3-0
        let gs = ring_graph(4);
        // Cut {0,1} | {2,3}
        // Edges crossing: 1-2 and 3-0
        // Biadjacency (rows=A={0,1}, cols=B={2,3}):
        //   0->{2,3}: [0, 1]  (0 connects to 3)
        //   1->{2,3}: [1, 0]  (1 connects to 2)
        // Rank = 2
        assert_eq!(gs.schmidt_rank_log2(&[0, 1]), 2);
    }

    #[test]
    fn test_linear_cluster4_rank() {
        // 0-1-2-3
        let gs = linear_cluster(4);
        // Cut {0,1} | {2,3}: only crossing edge is 1-2
        // Biadjacency:
        //   0->{2,3}: [0, 0]
        //   1->{2,3}: [1, 0]
        // Rank = 1
        assert_eq!(gs.schmidt_rank_log2(&[0, 1]), 1);
    }

    #[test]
    fn test_empty_partition_rank() {
        let gs = triangle_graph();
        assert_eq!(gs.schmidt_rank_log2(&[]), 0);
    }

    #[test]
    fn test_full_partition_rank() {
        let gs = triangle_graph();
        // All qubits in A, B is empty => rank 0
        assert_eq!(gs.schmidt_rank_log2(&[0, 1, 2]), 0);
    }

    #[test]
    fn test_entropy() {
        let mut gs = GraphState::empty(2);
        gs.toggle_edge(0, 1);
        let ent = gs.entanglement_entropy(&[0]);
        assert!((ent - std::f64::consts::LN_2).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Conversion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_zx_graph_vertex_edge_counts() {
        let gs = triangle_graph();
        let zx = gs.to_zx_graph();
        // 3 inputs + 3 spiders + 3 outputs = 9
        assert_eq!(zx.num_vertices(), 9);
        // 3 input-spider + 3 spider-output + 3 spider-spider = 9
        assert_eq!(zx.num_edges(), 9);
    }

    #[test]
    fn test_roundtrip_via_zx() {
        let gs = linear_cluster(4);
        let zx = gs.to_zx_graph();
        let gs2 = GraphState::from_zx_graph(&zx).unwrap();
        assert_eq!(gs, gs2);
    }

    #[test]
    fn test_roundtrip_via_zx_star() {
        let gs = star_graph(5);
        let zx = gs.to_zx_graph();
        let gs2 = GraphState::from_zx_graph(&zx).unwrap();
        assert_eq!(gs, gs2);
    }

    #[test]
    fn test_from_zx_rejects_x_spider() {
        let mut g = ZxGraph::new();
        let inp = g.add_vertex(VType::B);
        let x_spider = g.add_vertex(VType::X); // X spider, not Z
        let out = g.add_vertex(VType::B);
        g.add_edge(inp, x_spider);
        g.add_edge(x_spider, out);
        g.set_inputs(vec![inp]);
        g.set_outputs(vec![out]);

        let result = GraphState::from_zx_graph(&g);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_zx_rejects_mismatched_io_count() {
        let mut g = ZxGraph::new();
        let inp0 = g.add_vertex(VType::B);
        let spider = g.add_vertex(VType::Z);
        g.add_edge(inp0, spider);
        // One input, zero outputs
        g.set_inputs(vec![inp0]);
        g.set_outputs(vec![]);

        let result = GraphState::from_zx_graph(&g);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_zx_rejects_nonzero_phase() {
        let mut g = ZxGraph::new();
        let inp = g.add_vertex(VType::B);
        let spider = g.add_vertex(VType::Z);
        let out = g.add_vertex(VType::B);
        g.add_edge(inp, spider);
        g.add_edge(spider, out);
        g.set_phase(spider, (1_i64, 2_i64)); // pi/2 phase (half turns)
        g.set_inputs(vec![inp]);
        g.set_outputs(vec![out]);

        let result = GraphState::from_zx_graph(&g);
        assert!(result.is_err());
    }

    #[test]
    fn test_display() {
        let gs = GraphState::empty(2);
        let s = format!("{gs}");
        assert!(s.contains('0'));
    }
}
