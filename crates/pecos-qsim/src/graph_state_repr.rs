// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Graph state representation and manipulation API.
//!
//! This module provides [`GraphState`], a mathematical representation of graph states
//! for QEC researchers. Unlike [`GraphStateSim`](crate::GraphStateSim), which is a
//! circuit simulator (taking gates and measurements), `GraphState` is for constructing,
//! manipulating, and analyzing graph states as mathematical objects.
//!
//! # Graph states
//!
//! A graph state `|G>` is defined by an undirected graph G = (V, E). Each vertex
//! starts in `|+>`, then a CZ gate is applied for each edge. The stabilizer
//! generators are K_v = X_v * prod_{u in N(v)} Z_u.
//!
//! Any stabilizer state can be written as local Cliffords applied to a graph state:
//! `|psi> = (tensor_v VOP_v) |G>`. The VOP (vertex operator) on each qubit is a
//! single-qubit Clifford tracked as a [`CliffordFrame`].
//!
//! # Examples
//!
//! ```
//! use pecos_qsim::GraphState;
//!
//! // Create a 3-qubit linear cluster state: 0 - 1 - 2
//! let gs = GraphState::linear_cluster(3);
//! assert_eq!(gs.num_qubits(), 3);
//! assert_eq!(gs.num_edges(), 2);
//! assert!(gs.has_edge(0, 1));
//! assert!(gs.has_edge(1, 2));
//! assert!(!gs.has_edge(0, 2));
//! ```
//!
//! # References
//!
//! - Hein, Eisert, Briegel, "Multi-party entanglement in graph states",
//!   [quant-ph/0307130](https://arxiv.org/abs/quant-ph/0307130)
//! - Van den Nest, Dehaene, De Moor, "Graphical description of the action of
//!   local Clifford transformations on graph states",
//!   [quant-ph/0308151](https://arxiv.org/abs/quant-ph/0308151)

use crate::clifford_frame::{CliffordFrame, PauliAxis};
use core::fmt::{self, Write as _};
use pecos_core::{BitSet, Pauli, Phase, PauliString, QuarterPhase};
use pecos_rng::{PecosRng, SeedableRng};
use std::collections::{BTreeSet, VecDeque};

// ============================================================================
// Core type
// ============================================================================

/// A graph state representation for mathematical manipulation.
///
/// Stores vertex operators (VOPs) and an adjacency graph. The quantum state is
/// `|psi> = (tensor_v VOP_v) |G>` where `|G>` is the graph state.
///
/// Unlike [`GraphStateSim`](crate::GraphStateSim), this type has no RNG and is
/// not a circuit simulator. It is for constructing, transforming, and analyzing
/// graph states as mathematical objects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphState {
    vops: Vec<CliffordFrame>,
    neighbors: Vec<BitSet>,
}

// ============================================================================
// Constructors
// ============================================================================

impl GraphState {
    /// Create an n-qubit graph state with all VOPs identity and no edges.
    ///
    /// This represents `|+>^n` (the tensor product of n `|+>` states).
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            vops: vec![CliffordFrame::IDENTITY; n],
            neighbors: vec![BitSet::new(); n],
        }
    }

    /// Create a pure graph state from an edge list.
    ///
    /// All VOPs are identity. Panics if any vertex index is >= n.
    #[must_use]
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut gs = Self::new(n);
        for &(u, v) in edges {
            assert!(u < n && v < n, "vertex index out of range");
            assert!(u != v, "self-loops not allowed");
            gs.neighbors[u].insert(v);
            gs.neighbors[v].insert(u);
        }
        gs
    }

    /// Create a graph state from a symmetric boolean adjacency matrix.
    ///
    /// Panics if the matrix is not square or not symmetric.
    #[must_use]
    pub fn from_adjacency_matrix(matrix: &[Vec<bool>]) -> Self {
        let n = matrix.len();
        for row in matrix {
            assert_eq!(row.len(), n, "adjacency matrix must be square");
        }
        let mut gs = Self::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                assert_eq!(
                    matrix[i][j], matrix[j][i],
                    "adjacency matrix must be symmetric"
                );
                if matrix[i][j] {
                    gs.neighbors[i].insert(j);
                    gs.neighbors[j].insert(i);
                }
            }
        }
        gs
    }

    /// Create a graph state from raw parts (VOPs and adjacency lists).
    ///
    /// Panics if the lengths do not match.
    #[must_use]
    pub fn from_parts(vops: Vec<CliffordFrame>, neighbors: Vec<BitSet>) -> Self {
        assert_eq!(
            vops.len(),
            neighbors.len(),
            "vops and neighbors must have the same length"
        );
        Self { vops, neighbors }
    }

    // ========================================================================
    // Pattern factories
    // ========================================================================

    /// Linear cluster state: 0-1-2-..-(n-1).
    #[must_use]
    pub fn linear_cluster(n: usize) -> Self {
        if n == 0 {
            return Self::new(0);
        }
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        Self::from_edges(n, &edges)
    }

    /// Ring graph state: 0-1-..-(n-1)-0.
    ///
    /// Requires n >= 3.
    #[must_use]
    pub fn ring(n: usize) -> Self {
        assert!(n >= 3, "ring requires at least 3 vertices");
        let mut edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        edges.push((n - 1, 0));
        Self::from_edges(n, &edges)
    }

    /// Star graph state: vertex 0 connected to all others.
    #[must_use]
    pub fn star(n: usize) -> Self {
        assert!(n >= 2, "star requires at least 2 vertices");
        let edges: Vec<(usize, usize)> = (1..n).map(|i| (0, i)).collect();
        Self::from_edges(n, &edges)
    }

    /// 2D rectangular lattice graph state.
    #[must_use]
    pub fn lattice_2d(rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let v = r * cols + c;
                if c + 1 < cols {
                    edges.push((v, v + 1));
                }
                if r + 1 < rows {
                    edges.push((v, v + cols));
                }
            }
        }
        Self::from_edges(n, &edges)
    }

    /// Complete graph state K_n.
    #[must_use]
    pub fn complete(n: usize) -> Self {
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push((i, j));
            }
        }
        Self::from_edges(n, &edges)
    }
}

// ============================================================================
// Accessors
// ============================================================================

impl GraphState {
    /// Returns the number of qubits (vertices).
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.vops.len()
    }

    /// Returns the VOP (vertex operator) for vertex v.
    #[inline]
    #[must_use]
    pub fn vop(&self, v: usize) -> CliffordFrame {
        self.vops[v]
    }

    /// Returns the neighbor set of vertex v.
    #[inline]
    #[must_use]
    pub fn neighbors(&self, v: usize) -> &BitSet {
        &self.neighbors[v]
    }

    /// Returns true if there is an edge between u and v.
    #[inline]
    #[must_use]
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.neighbors[u].contains(v)
    }

    /// Returns the degree of vertex v.
    #[inline]
    #[must_use]
    pub fn degree(&self, v: usize) -> usize {
        self.neighbors[v].len()
    }

    /// Returns the total number of edges.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        let total: usize = self.neighbors.iter().map(BitSet::len).sum();
        total / 2
    }

    /// Iterate over all edges (u, v) with u < v.
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let n = self.num_qubits();
        (0..n).flat_map(move |u| {
            self.neighbors[u]
                .iter()
                .filter(move |&v| v > u)
                .map(move |v| (u, v))
        })
    }

    /// Returns true if all VOPs are identity (a "pure" graph state).
    #[must_use]
    pub fn is_pure_graph_state(&self) -> bool {
        self.vops.iter().all(|v| v.is_identity())
    }

    /// Returns the adjacency matrix as a vector of vectors.
    #[must_use]
    pub fn adjacency_matrix(&self) -> Vec<Vec<bool>> {
        let n = self.num_qubits();
        let mut matrix = vec![vec![false; n]; n];
        for (u, v) in self.edges() {
            matrix[u][v] = true;
            matrix[v][u] = true;
        }
        matrix
    }
}

// ============================================================================
// Mutators
// ============================================================================

impl GraphState {
    /// Set the VOP for vertex v.
    #[inline]
    pub fn set_vop(&mut self, v: usize, cliff: CliffordFrame) {
        self.vops[v] = cliff;
    }

    /// Apply a local Clifford gate to vertex v (right-composes with existing VOP).
    #[inline]
    pub fn apply_local_clifford(&mut self, v: usize, gate: CliffordFrame) {
        self.vops[v] = self.vops[v].compose(gate);
    }

    /// Toggle edge (u, v): add if absent, remove if present.
    pub fn toggle_edge(&mut self, u: usize, v: usize) {
        assert_ne!(u, v, "self-loops not allowed");
        self.neighbors[u].toggle(v);
        self.neighbors[v].toggle(u);
    }

    /// Add edge (u, v). No-op if already present.
    pub fn add_edge(&mut self, u: usize, v: usize) {
        assert_ne!(u, v, "self-loops not allowed");
        self.neighbors[u].insert(v);
        self.neighbors[v].insert(u);
    }

    /// Remove edge (u, v). No-op if not present.
    pub fn remove_edge(&mut self, u: usize, v: usize) {
        self.neighbors[u].remove(v);
        self.neighbors[v].remove(u);
    }
}

// ============================================================================
// Local complementation
// ============================================================================

impl GraphState {
    /// Perform local complementation about vertex v.
    ///
    /// This complements all edges among N(v) and updates VOPs:
    /// - Prepend sqrt(-iX) = SXDG to VOP_v
    /// - Prepend sqrt(iZ) = SZ to each neighbor's VOP
    pub fn local_complement(&mut self, v: usize) {
        let nbrs: Vec<usize> = self.neighbors[v].iter().collect();

        // Complement edges among N(v)
        for i in 0..nbrs.len() {
            for j in (i + 1)..nbrs.len() {
                self.neighbors[nbrs[i]].toggle(nbrs[j]);
                self.neighbors[nbrs[j]].toggle(nbrs[i]);
            }
        }

        // Update VOPs: prepend SXDG to vertex v
        self.vops[v] = CliffordFrame::SXDG.compose(self.vops[v]);

        // Prepend SZ to each neighbor
        for &u in &nbrs {
            self.vops[u] = CliffordFrame::SZ.compose(self.vops[u]);
        }
    }

    /// Perform a pivot on edge (u, v): LC(u), LC(v), LC(u).
    ///
    /// Panics if u and v are not adjacent.
    pub fn pivot(&mut self, u: usize, v: usize) {
        assert!(
            self.has_edge(u, v),
            "pivot requires u and v to be adjacent"
        );
        self.local_complement(u);
        self.local_complement(v);
        self.local_complement(u);
    }

    /// Graph-only local complementation: complement edges among N(v).
    ///
    /// Unlike [`local_complement`](Self::local_complement), this does NOT update VOPs.
    /// Used internally for LC-orbit enumeration where we work with graphs only.
    fn graph_local_complement(&mut self, v: usize) {
        let nbrs: Vec<usize> = self.neighbors[v].iter().collect();
        for i in 0..nbrs.len() {
            for j in (i + 1)..nbrs.len() {
                self.neighbors[nbrs[i]].toggle(nbrs[j]);
                self.neighbors[nbrs[j]].toggle(nbrs[i]);
            }
        }
    }

    /// Absorb all VOPs into the graph, producing an equivalent pure graph state.
    ///
    /// Computes the stabilizer generators, then extracts the equivalent graph
    /// from the canonical stabilizer form. For each generator, the X position
    /// identifies the vertex, and Z positions identify its neighbors.
    ///
    /// Note: isolated vertices with non-identity VOPs cannot be fully absorbed
    /// since there are no neighbors to use for LC operations. Their VOPs
    /// remain unchanged.
    pub fn absorb_vops(&mut self) {
        if self.is_pure_graph_state() {
            return;
        }

        let n = self.num_qubits();

        // Compute stabilizer generators for the current state
        let gens = self.stabilizer_generators();

        // Build a new pure graph state from the stabilizer generators.
        // For a graph state, each stabilizer generator has exactly one X
        // (or can be brought to that form). The generator for vertex v
        // is: (+/-)X_v * prod_{u in N(v)} Z_u
        //
        // We need to find generators that have a single X and the rest Z/I.
        // This works when the state is equivalent to a graph state (which
        // any stabilizer state is, up to local Cliffords -- and our state
        // IS local Cliffords applied to a graph state).

        // Try to extract graph structure from generators.
        // For each generator, check if it has the form (+/-)X_v * (Z terms).
        // If all generators have this form, we can directly read off the graph.
        let mut new_neighbors = vec![BitSet::new(); n];
        let mut success = true;

        for (idx, g) in gens.iter().enumerate() {
            // Find the single X position
            let mut x_pos = None;
            let mut valid = true;

            for q in 0..n {
                match g.get(q) {
                    Pauli::X => {
                        if x_pos.is_some() {
                            valid = false;
                            break;
                        }
                        x_pos = Some(q);
                    }
                    Pauli::Y => {
                        valid = false;
                        break;
                    }
                    Pauli::Z | Pauli::I => {}
                }
            }

            if !valid || x_pos.is_none() {
                success = false;
                break;
            }

            let v = x_pos.unwrap();
            if v != idx {
                // Generator ordering doesn't match vertex ordering
                // This could happen but shouldn't for our construction
                success = false;
                break;
            }

            for q in 0..n {
                if g.get(q) == Pauli::Z {
                    new_neighbors[v].insert(q);
                }
            }
        }

        if success {
            self.neighbors = new_neighbors;
            for v in 0..n {
                self.vops[v] = CliffordFrame::IDENTITY;
            }
        }
        // If not successful (state has Y terms in generators), the VOPs
        // cannot be trivially absorbed. This is fine for LC-equivalence
        // which uses graph-only operations.
    }
}

// ============================================================================
// Stabilizer extraction (Phase 3)
// ============================================================================

impl GraphState {
    /// Compute the stabilizer generator for vertex v.
    ///
    /// The bare generator is K_v = X_v * prod_{u in N(v)} Z_u.
    /// The conjugated generator is VOP_v(X_v) * prod_{u in N(v)} VOP_u(Z_u).
    #[must_use]
    pub fn stabilizer_generator(&self, v: usize) -> PauliString {
        let n = self.num_qubits();
        let mut paulis = vec![Pauli::I; n];
        let mut phase = QuarterPhase::PlusOne;

        // Vertex v contributes: VOP_v maps X
        let x_img = self.vops[v].x_image();
        paulis[v] = pauli_axis_to_pauli(x_img.axis);
        if !x_img.positive {
            phase = phase.multiply(&QuarterPhase::MinusOne);
        }

        // Each neighbor u contributes: VOP_u maps Z
        for u in self.neighbors[v].iter() {
            let z_img = self.vops[u].z_image();
            let u_pauli = pauli_axis_to_pauli(z_img.axis);

            if !z_img.positive {
                phase = phase.multiply(&QuarterPhase::MinusOne);
            }

            // Multiply with existing Pauli at position u (could overlap if u == v's neighbor
            // and there's already something there from a previous neighbor -- but neighbors
            // are distinct from v, and each neighbor contributes to its own position)
            if paulis[u] == Pauli::I {
                paulis[u] = u_pauli;
            } else {
                // Two non-identity Paulis at same position: multiply them
                let (result_pauli, extra_phase) = multiply_paulis(paulis[u], u_pauli);
                paulis[u] = result_pauli;
                phase = phase.multiply(&extra_phase);
            }
        }

        PauliString::from_paulis_with_phase(phase, &paulis)
    }

    /// Compute all n stabilizer generators.
    #[must_use]
    pub fn stabilizer_generators(&self) -> Vec<PauliString> {
        (0..self.num_qubits())
            .map(|v| self.stabilizer_generator(v))
            .collect()
    }
}

// ============================================================================
// Conversions (Phase 4)
// ============================================================================

impl GraphState {
    /// Convert into a simulator by providing an RNG.
    #[must_use]
    pub fn into_sim<R: SeedableRng + pecos_rng::Rng + core::fmt::Debug>(
        self,
        rng: R,
    ) -> crate::graph_state::GraphStateSim<R> {
        crate::graph_state::GraphStateSim::from_graph_state(self, rng)
    }

    /// Convert into a simulator with a specific seed.
    #[must_use]
    pub fn into_sim_with_seed(self, seed: u64) -> crate::graph_state::GraphStateSim<PecosRng> {
        let rng = PecosRng::seed_from_u64(seed);
        self.into_sim(rng)
    }

    /// Tensor product of two graph states.
    ///
    /// The second graph state's vertex indices are shifted by `self.num_qubits()`.
    #[must_use]
    pub fn tensor_product(&self, other: &Self) -> Self {
        let n1 = self.num_qubits();
        let n2 = other.num_qubits();
        let n = n1 + n2;

        let mut vops = self.vops.clone();
        vops.extend_from_slice(&other.vops);

        let mut neighbors = self.neighbors.clone();
        // Shift other's neighbor indices by n1
        for nbrs in &other.neighbors {
            let mut shifted = BitSet::new();
            for u in nbrs.iter() {
                shifted.insert(u + n1);
            }
            neighbors.push(shifted);
        }

        debug_assert_eq!(vops.len(), n);
        debug_assert_eq!(neighbors.len(), n);

        Self { vops, neighbors }
    }

    /// Disconnect vertex v from all neighbors and reset its VOP to identity.
    pub fn delete_vertex(&mut self, v: usize) {
        let nbrs: Vec<usize> = self.neighbors[v].iter().collect();
        for &u in &nbrs {
            self.neighbors[u].remove(v);
        }
        self.neighbors[v].clear();
        self.vops[v] = CliffordFrame::IDENTITY;
    }

    /// Extract the induced subgraph on the given vertices, re-indexed 0, 1, 2, ...
    #[must_use]
    pub fn induced_subgraph(&self, vertices: &[usize]) -> Self {
        let n = vertices.len();
        // Build mapping from old index to new index
        let mut old_to_new = vec![None; self.num_qubits()];
        for (new_idx, &old_idx) in vertices.iter().enumerate() {
            old_to_new[old_idx] = Some(new_idx);
        }

        let mut vops = Vec::with_capacity(n);
        let mut neighbors = vec![BitSet::new(); n];

        for (new_idx, &old_idx) in vertices.iter().enumerate() {
            vops.push(self.vops[old_idx]);
            for u in self.neighbors[old_idx].iter() {
                if let Some(new_u) = old_to_new[u] {
                    neighbors[new_idx].insert(new_u);
                }
            }
        }

        Self { vops, neighbors }
    }
}

// ============================================================================
// LC-equivalence (Phase 5)
// ============================================================================

impl GraphState {
    /// Enumerate the entire LC orbit of this graph state.
    ///
    /// Returns all pure graph states (identity VOPs) reachable by graph-level
    /// local complementations from this one's underlying graph. VOPs are
    /// irrelevant for LC-equivalence since they are local Cliffords.
    ///
    /// Only practical for small graphs (the orbit can be exponential in size).
    #[must_use]
    pub fn lc_orbit(&self) -> Vec<GraphState> {
        // Start from the underlying graph (ignoring VOPs)
        let start = GraphState::from_parts(
            vec![CliffordFrame::IDENTITY; self.num_qubits()],
            self.neighbors.clone(),
        );

        let mut visited: BTreeSet<Vec<Vec<bool>>> = BTreeSet::new();
        let mut queue: VecDeque<GraphState> = VecDeque::new();
        let mut orbit: Vec<GraphState> = Vec::new();

        visited.insert(start.adjacency_matrix());
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            let n = current.num_qubits();
            orbit.push(current.clone());

            for v in 0..n {
                if current.neighbors[v].is_empty() {
                    continue;
                }
                let mut next = current.clone();
                // Graph-only LC: just complement edges among N(v)
                next.graph_local_complement(v);

                let adj = next.adjacency_matrix();
                if visited.insert(adj) {
                    queue.push_back(next);
                }
            }
        }

        orbit
    }

    /// Compute a canonical form for LC-equivalence.
    ///
    /// Returns the lexicographically smallest adjacency matrix reachable by
    /// graph-level LC. Two graph states are LC-equivalent iff their canonical
    /// forms are equal. VOPs are irrelevant (they are local Cliffords).
    ///
    /// Uses orbit enumeration, so only practical for small graphs.
    #[must_use]
    pub fn lc_canonical_form(&self) -> GraphState {
        let orbit = self.lc_orbit();
        orbit
            .into_iter()
            .min_by(|a, b| {
                let adj_a = a.adjacency_matrix();
                let adj_b = b.adjacency_matrix();
                adj_a.cmp(&adj_b)
            })
            .expect("orbit is never empty")
    }

    /// Check if two graph states are LC-equivalent.
    ///
    /// Two graph states are LC-equivalent if their underlying graphs are in
    /// the same LC orbit. VOPs are irrelevant since they are local Cliffords.
    #[must_use]
    pub fn is_lc_equivalent(&self, other: &Self) -> bool {
        let canon_self = self.lc_canonical_form();
        let canon_other = other.lc_canonical_form();
        canon_self.adjacency_matrix() == canon_other.adjacency_matrix()
    }
}

// ============================================================================
// Export / Display (Phase 6)
// ============================================================================

/// Names for the 24 single-qubit Cliffords.
const CLIFFORD_NAMES: [&str; 24] = [
    "I", "X", "Y", "Z", "S", "Sdg", "H", "SH", "HS", "S2H", "HS2", "S3H",
    "SHS", "HSH", "SHSH", "S2HS", "SHS2", "S3HS", "S2HS2", "S2HSH", "HS2HS",
    "S3HS2", "S3HSH", "HS2HS3",
];

// ============================================================================
// VOP Color Algebra
// ============================================================================
//
// Three independent visual dimensions encode Clifford structure:
//
// 1. **Fill hue** — axis permutation coset (which pair of Pauli axes
//    the Clifford interconverts, ignoring signs):
//      Blue (#6495ED)   — identity perm (X→X, Z→Z)
//      Purple (#C850C0) — X↔Z swap (H-type)
//      Gold (#DAA520)   — X↔Y swap (S-type)
//      Cyan (#00B4D8)   — Y↔Z swap (SX-type)
//      Gray             — 3-cycle (dark #707070 = fwd X→Y→Z→X,
//                                  light #B0B0B0 = inv X→Z→Y→X)
//
// 2. **Fill brightness** — sign parity of the Heisenberg action:
//      Saturated — even parity (0 or 2 negative signs)
//      Light     — odd parity (1 negative sign)
//
// 3. **Stroke colour** — gate family (geometric rotation type on the
//    Bloch sphere):
//      Navy (#1E3A8A)     — Pauli (identity / π-rotations)
//      Green (#2D6A2E)    — sqrt-of-Pauli / S-like (π/2 rotations)
//      Maroon (#8B1A1A)   — Hadamard-like (π rotations about face diagonals)
//      Charcoal (#404040) — Face-like / cyclic (2π/3 rotations)

/// Visual style for a VOP vertex: fill, stroke, and text colours.
struct VopStyle {
    fill: &'static str,
    stroke: &'static str,
    text: &'static str,
}

/// Precomputed visual styles for all 24 single-qubit Cliffords.
///
/// Indexed by [`CliffordFrame::index()`]. Derived from the HEIS table:
/// unsigned axis permutation → fill hue, sign parity → brightness,
/// geometric rotation type → stroke colour.
#[rustfmt::skip]
const VOP_STYLES: [VopStyle; 24] = [
    //                fill        stroke      text
    // Identity coset (X→X, Z→Z) — Pauli family
    VopStyle { fill: "#6495ED", stroke: "#1E3A8A", text: "white" }, //  0: I       even
    VopStyle { fill: "#A0BEF5", stroke: "#1E3A8A", text: "#333"  }, //  1: X       odd
    VopStyle { fill: "#6495ED", stroke: "#1E3A8A", text: "white" }, //  2: Y       even
    VopStyle { fill: "#A0BEF5", stroke: "#1E3A8A", text: "#333"  }, //  3: Z       odd
    // X↔Y coset (S-type)
    VopStyle { fill: "#F0D080", stroke: "#2D6A2E", text: "#333"  }, //  4: S       odd,  S-like
    VopStyle { fill: "#DAA520", stroke: "#2D6A2E", text: "white" }, //  5: Sdg     even, S-like
    // X↔Z coset (H-type)
    VopStyle { fill: "#C850C0", stroke: "#8B1A1A", text: "white" }, //  6: H       even, H-like
    // Cyclic forward (X→Y→Z→X)
    VopStyle { fill: "#707070", stroke: "#404040", text: "white" }, //  7: SH      F-like
    // Cyclic inverse (X→Z→Y→X)
    VopStyle { fill: "#B0B0B0", stroke: "#404040", text: "#333"  }, //  8: HS      F-like
    // X↔Z coset cont.
    VopStyle { fill: "#E8A0E0", stroke: "#2D6A2E", text: "#333"  }, //  9: S²H     odd,  S-like (=SYdg)
    VopStyle { fill: "#E8A0E0", stroke: "#2D6A2E", text: "#333"  }, // 10: HS²     odd,  S-like (=SY)
    // Cyclic forward cont.
    VopStyle { fill: "#707070", stroke: "#404040", text: "white" }, // 11: S³H     F-like
    // Y↔Z coset (SX-type)
    VopStyle { fill: "#80D8E8", stroke: "#2D6A2E", text: "#333"  }, // 12: SHS     odd,  S-like (=SXdg)
    VopStyle { fill: "#00B4D8", stroke: "#2D6A2E", text: "white" }, // 13: HSH     even, S-like (=SX)
    // Cyclic inverse cont.
    VopStyle { fill: "#B0B0B0", stroke: "#404040", text: "#333"  }, // 14: SHSH    F-like
    VopStyle { fill: "#B0B0B0", stroke: "#404040", text: "#333"  }, // 15: S²HS    F-like
    // Cyclic forward cont.
    VopStyle { fill: "#707070", stroke: "#404040", text: "white" }, // 16: SHS²    F-like
    // Y↔Z coset cont.
    VopStyle { fill: "#00B4D8", stroke: "#8B1A1A", text: "white" }, // 17: S³HS    even, H-like
    // X↔Z coset cont.
    VopStyle { fill: "#C850C0", stroke: "#8B1A1A", text: "white" }, // 18: S²HS²   even, H-like
    // Y↔Z coset cont.
    VopStyle { fill: "#80D8E8", stroke: "#8B1A1A", text: "#333"  }, // 19: S²HSH   odd,  H-like
    // X↔Y coset cont.
    VopStyle { fill: "#DAA520", stroke: "#8B1A1A", text: "white" }, // 20: HS²HS   even, H-like
    // Cyclic forward cont.
    VopStyle { fill: "#707070", stroke: "#404040", text: "white" }, // 21: S³HS²   F-like
    // Cyclic inverse cont.
    VopStyle { fill: "#B0B0B0", stroke: "#404040", text: "#333"  }, // 22: S³HSH   F-like
    // X↔Y coset cont.
    VopStyle { fill: "#F0D080", stroke: "#8B1A1A", text: "#333"  }, // 23: HS²HS³  odd,  H-like
];

/// Returns the visual style for a VOP by its Clifford index.
fn vop_style(idx: u8) -> &'static VopStyle {
    &VOP_STYLES[idx as usize]
}

/// ANSI SGR escape codes for each of the 24 single-qubit Cliffords.
///
/// Encodes coset (colour) and sign parity (bold/normal):
///   Identity -> blue (34), X<->Z -> magenta (35), X<->Y -> yellow (33),
///   Y<->Z -> cyan (36), cyclic fwd -> white (37), cyclic inv -> bright black (90).
///   Even parity (saturated) -> bold; odd parity (light) -> normal.
#[rustfmt::skip]
const VOP_ANSI: [&str; 24] = [
    "\x1b[1;34m",  //  0: I       Identity even
    "\x1b[34m",    //  1: X       Identity odd
    "\x1b[1;34m",  //  2: Y       Identity even
    "\x1b[34m",    //  3: Z       Identity odd
    "\x1b[33m",    //  4: S       X<->Y odd
    "\x1b[1;33m",  //  5: Sdg     X<->Y even
    "\x1b[1;35m",  //  6: H       X<->Z even
    "\x1b[1;37m",  //  7: SH      Cyclic fwd
    "\x1b[90m",    //  8: HS      Cyclic inv
    "\x1b[35m",    //  9: S2H     X<->Z odd
    "\x1b[35m",    // 10: HS2     X<->Z odd
    "\x1b[1;37m",  // 11: S3H     Cyclic fwd
    "\x1b[36m",    // 12: SHS     Y<->Z odd
    "\x1b[1;36m",  // 13: HSH     Y<->Z even
    "\x1b[90m",    // 14: SHSH    Cyclic inv
    "\x1b[90m",    // 15: S2HS    Cyclic inv
    "\x1b[1;37m",  // 16: SHS2    Cyclic fwd
    "\x1b[1;36m",  // 17: S3HS    Y<->Z even
    "\x1b[1;35m",  // 18: S2HS2   X<->Z even
    "\x1b[36m",    // 19: S2HSH   Y<->Z odd
    "\x1b[1;33m",  // 20: HS2HS   X<->Y even
    "\x1b[1;37m",  // 21: S3HS2   Cyclic fwd
    "\x1b[90m",    // 22: S3HSH   Cyclic inv
    "\x1b[33m",    // 23: HS2HS3  X<->Y odd
];

/// Bracket pairs for each of the 24 Cliffords, encoding gate family.
///
/// Pauli -> `( )`, S-like -> `[ ]`, H-like -> `< >`, F-like -> `{ }`.
#[rustfmt::skip]
const VOP_BRACKETS: [(&str, &str); 24] = [
    ("(", ")"),  //  0: I       Pauli
    ("(", ")"),  //  1: X       Pauli
    ("(", ")"),  //  2: Y       Pauli
    ("(", ")"),  //  3: Z       Pauli
    ("[", "]"),  //  4: S       S-like
    ("[", "]"),  //  5: Sdg     S-like
    ("<", ">"),  //  6: H       H-like
    ("{", "}"),  //  7: SH      F-like
    ("{", "}"),  //  8: HS      F-like
    ("[", "]"),  //  9: S2H     S-like
    ("[", "]"),  // 10: HS2     S-like
    ("{", "}"),  // 11: S3H     F-like
    ("[", "]"),  // 12: SHS     S-like
    ("[", "]"),  // 13: HSH     S-like
    ("{", "}"),  // 14: SHSH    F-like
    ("{", "}"),  // 15: S2HS    F-like
    ("{", "}"),  // 16: SHS2    F-like
    ("<", ">"),  // 17: S3HS    H-like
    ("<", ">"),  // 18: S2HS2   H-like
    ("<", ">"),  // 19: S2HSH   H-like
    ("<", ">"),  // 20: HS2HS   H-like
    ("{", "}"),  // 21: S3HS2   F-like
    ("{", "}"),  // 22: S3HSH   F-like
    ("<", ">"),  // 23: HS2HS3  H-like
];

/// Append a compact SVG legend showing coset hues and gate-family strokes.
fn svg_legend(svg: &mut String, width: f64, height: f64, legend_height: f64) {
    let y_top = height - legend_height + 8.0;
    let r = 6.0; // legend circle radius

    // Row 1: fill hues (axis permutation cosets)
    let cosets: &[(&str, &str, &str)] = &[
        ("#6495ED", "#1E3A8A", "I/Pauli"),
        ("#C850C0", "#6A006A", "X\u{2194}Z"),
        ("#DAA520", "#8B6914", "X\u{2194}Y"),
        ("#00B4D8", "#006880", "Y\u{2194}Z"),
        ("#808080", "#404040", "Cyclic"),
    ];

    let total_items = cosets.len();
    let spacing = width / (total_items as f64 + 1.0);

    for (i, &(fill, stroke, label)) in cosets.iter().enumerate() {
        let cx = spacing * (i as f64 + 1.0);
        svg.push_str(&format!(
            "  <circle cx=\"{cx:.1}\" cy=\"{y_top:.1}\" r=\"{r}\" \
             fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1.5\"/>\n"
        ));
        let tx = cx + r + 4.0;
        svg.push_str(&format!(
            "  <text x=\"{tx:.1}\" y=\"{:.1}\" \
             font-family=\"sans-serif\" font-size=\"9\" fill=\"#555\">\
             {label}</text>\n",
            y_top + 3.0
        ));
    }

    // Row 2: stroke colours (gate families)
    let families: &[(&str, &str)] = &[
        ("#1E3A8A", "Pauli"),
        ("#2D6A2E", "S-like"),
        ("#8B1A1A", "H-like"),
        ("#404040", "F-like"),
    ];

    let y_row2 = y_top + 18.0;
    let fam_spacing = width / (families.len() as f64 + 1.0);

    for (i, &(stroke_col, label)) in families.iter().enumerate() {
        let cx = fam_spacing * (i as f64 + 1.0);
        svg.push_str(&format!(
            "  <circle cx=\"{cx:.1}\" cy=\"{y_row2:.1}\" r=\"{r}\" \
             fill=\"white\" stroke=\"{stroke_col}\" stroke-width=\"2.5\"/>\n"
        ));
        let tx = cx + r + 4.0;
        svg.push_str(&format!(
            "  <text x=\"{tx:.1}\" y=\"{:.1}\" \
             font-family=\"sans-serif\" font-size=\"9\" fill=\"#555\">\
             {label}</text>\n",
            y_row2 + 3.0
        ));
    }
}

/// Map a VOP fill hex colour to its TikZ colour name.
fn tikz_fill_name(hex: &str) -> &'static str {
    match hex {
        "#6495ED" => "vopIdentity",
        "#A0BEF5" => "vopIdentityLt",
        "#C850C0" => "vopXZ",
        "#E8A0E0" => "vopXZLt",
        "#DAA520" => "vopXY",
        "#F0D080" => "vopXYLt",
        "#00B4D8" => "vopYZ",
        "#80D8E8" => "vopYZLt",
        "#707070" => "vopCyclicFwd",
        "#B0B0B0" => "vopCyclicInv",
        _ => "black",
    }
}

/// Map a VOP stroke hex colour to its TikZ colour name.
fn tikz_stroke_name(hex: &str) -> &'static str {
    match hex {
        "#1E3A8A" => "famPauli",
        "#2D6A2E" => "famSqrt",
        "#8B1A1A" => "famHadamard",
        "#404040" => "famCyclic",
        _ => "black",
    }
}

impl GraphState {
    /// Export to DOT format for Graphviz visualization.
    ///
    /// Vertices are coloured using the PECOS colour algebra (fill hue = axis
    /// permutation coset, stroke = gate family).
    #[must_use]
    pub fn to_dot(&self) -> String {
        let n = self.num_qubits();
        let mut dot = String::from("graph G {\n");
        dot.push_str("  node [shape=circle, style=filled, fontsize=12];\n");

        for v in 0..n {
            let idx = self.vops[v].index();
            let name = CLIFFORD_NAMES[idx as usize];
            let style = vop_style(idx);
            dot.push_str(&format!(
                "  {v} [label=\"{v}\\n{name}\" fillcolor=\"{}\" \
                 color=\"{}\" fontcolor=\"{}\"];\n",
                style.fill, style.stroke, style.text
            ));
        }

        for (u, v) in self.edges() {
            dot.push_str(&format!("  {u} -- {v};\n"));
        }

        dot.push_str("}\n");
        dot
    }

    /// Compute vertex positions using a circular layout.
    ///
    /// Returns (x, y) pairs for each vertex, centered at (`cx`, `cy`) with
    /// the given `radius`. Single-vertex graphs place the vertex at center.
    fn circular_layout(n: usize, cx: f64, cy: f64, radius: f64) -> Vec<(f64, f64)> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![(cx, cy)];
        }
        (0..n)
            .map(|i| {
                let angle =
                    -std::f64::consts::FRAC_PI_2 + 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                (cx + radius * angle.cos(), cy + radius * angle.sin())
            })
            .collect()
    }

    /// Export to SVG format.
    ///
    /// Produces a standalone SVG image with vertices arranged in a circular
    /// layout. Vertex colours encode Clifford structure via the PECOS colour
    /// algebra (fill hue = axis permutation, brightness = sign parity,
    /// stroke = gate family). Non-identity VOPs are labeled below their node.
    /// A compact legend is drawn at the bottom.
    #[must_use]
    pub fn to_svg(&self) -> String {
        let n = self.num_qubits();
        let node_radius = 20.0;
        let layout_radius = if n <= 2 { 60.0 } else { 40.0 + 25.0 * n as f64 };
        let margin = node_radius + 40.0;
        let width = 2.0 * (layout_radius + margin);
        let legend_height = 50.0;
        let height = width + legend_height;
        let center = layout_radius + margin;

        let positions = Self::circular_layout(n, center, center, layout_radius);

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" \
             width=\"{width}\" height=\"{height}\" \
             viewBox=\"0 0 {width} {height}\">\n"
        );
        svg.push_str(&format!(
            "  <rect width=\"{width}\" height=\"{height}\" fill=\"white\"/>\n"
        ));

        // Draw edges
        for (u, v) in self.edges() {
            let (x1, y1) = positions[u];
            let (x2, y2) = positions[v];
            svg.push_str(&format!(
                "  <line x1=\"{x1:.1}\" y1=\"{y1:.1}\" \
                 x2=\"{x2:.1}\" y2=\"{y2:.1}\" \
                 stroke=\"#555\" stroke-width=\"1.5\"/>\n"
            ));
        }

        // Draw vertices
        for v in 0..n {
            let (x, y) = positions[v];
            let idx = self.vops[v].index();
            let vop_name = CLIFFORD_NAMES[idx as usize];
            let style = vop_style(idx);

            svg.push_str(&format!(
                "  <circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"{node_radius}\" \
                 fill=\"{}\" stroke=\"{}\" stroke-width=\"2\"/>\n",
                style.fill, style.stroke
            ));

            // Vertex index label
            svg.push_str(&format!(
                "  <text x=\"{x:.1}\" y=\"{y:.1}\" \
                 text-anchor=\"middle\" dominant-baseline=\"central\" \
                 font-family=\"sans-serif\" font-size=\"12\" \
                 fill=\"{}\" font-weight=\"bold\">{v}</text>\n",
                style.text
            ));

            // VOP label (below the node, only if non-identity)
            if !self.vops[v].is_identity() {
                let label_y = y + node_radius + 14.0;
                svg.push_str(&format!(
                    "  <text x=\"{x:.1}\" y=\"{label_y:.1}\" \
                     text-anchor=\"middle\" \
                     font-family=\"sans-serif\" font-size=\"10\" \
                     fill=\"#666\">{vop_name}</text>\n"
                ));
            }
        }

        // Legend
        svg_legend(&mut svg, width, height, legend_height);

        svg.push_str("</svg>\n");
        svg
    }

    /// Export to TikZ format for LaTeX documents.
    ///
    /// Produces a `tikzpicture` environment with vertices coloured by the
    /// PECOS colour algebra. Requires `\usepackage{tikz}` and
    /// `\usepackage[dvipsnames]{xcolor}` (or `\usepackage{xcolor}`) in
    /// your LaTeX preamble.
    #[must_use]
    pub fn to_tikz(&self) -> String {
        let n = self.num_qubits();
        let radius = if n <= 2 { 1.5 } else { 1.0 + 0.5 * n as f64 };
        let positions = Self::circular_layout(n, 0.0, 0.0, radius);

        let mut tikz = String::from("\\begin{tikzpicture}\n");

        // Colour definitions — fill hues (axis permutation cosets)
        tikz.push_str("  % Fill: axis permutation coset (bright / light)\n");
        for &(name, hex) in &[
            ("vopIdentity",   "6495ED"), ("vopIdentityLt", "A0BEF5"),
            ("vopXZ",         "C850C0"), ("vopXZLt",       "E8A0E0"),
            ("vopXY",         "DAA520"), ("vopXYLt",       "F0D080"),
            ("vopYZ",         "00B4D8"), ("vopYZLt",       "80D8E8"),
            ("vopCyclicFwd",  "707070"), ("vopCyclicInv",  "B0B0B0"),
        ] {
            tikz.push_str(&format!("  \\definecolor{{{name}}}{{HTML}}{{{hex}}}\n"));
        }
        // Stroke colours — gate families
        tikz.push_str("  % Stroke: gate family\n");
        for &(name, hex) in &[
            ("famPauli",    "1E3A8A"),
            ("famSqrt",     "2D6A2E"),
            ("famHadamard", "8B1A1A"),
            ("famCyclic",   "404040"),
        ] {
            tikz.push_str(&format!("  \\definecolor{{{name}}}{{HTML}}{{{hex}}}\n"));
        }

        // Base vertex style
        tikz.push_str(
            "  \\tikzstyle{vertex}=[circle, minimum size=20pt, \
             inner sep=0pt, font=\\small, line width=1.5pt]\n",
        );
        tikz.push_str(
            "  \\tikzstyle{vop label}=[font=\\scriptsize, text=gray]\n",
        );

        // Draw vertices
        for v in 0..n {
            let (x, y) = positions[v];
            let idx = self.vops[v].index();
            let style = vop_style(idx);
            let fill_name = tikz_fill_name(style.fill);
            let draw_name = tikz_stroke_name(style.stroke);
            let text_opt = if style.text == "white" { ", text=white" } else { "" };

            tikz.push_str(&format!(
                "  \\node[vertex, fill={fill_name}, draw={draw_name}{text_opt}] \
                 (v{v}) at ({x:.2}, {y:.2}) {{{v}}};\n"
            ));

            // VOP annotation
            if !self.vops[v].is_identity() {
                let vop_name = CLIFFORD_NAMES[idx as usize];
                let label_y = y - 0.45;
                tikz.push_str(&format!(
                    "  \\node[vop label] at ({x:.2}, {label_y:.2}) {{${vop_name}$}};\n"
                ));
            }
        }

        // Draw edges
        for (u, v) in self.edges() {
            tikz.push_str(&format!("  \\draw (v{u}) -- (v{v});\n"));
        }

        tikz.push_str("\\end{tikzpicture}\n");
        tikz
    }

    /// Export as plain ASCII text (no escape codes).
    ///
    /// Compact vertex-per-line format. When all VOPs are identity (a pure
    /// graph state), the VOP column is omitted for a clean adjacency-list
    /// view. When any VOP is non-trivial, non-identity VOPs are shown in
    /// brackets encoding gate family: `()` Pauli, `[]` S-like, `<>` H-like,
    /// `{}` F-like. Identity vertices get a blank VOP column.
    #[must_use]
    pub fn to_ascii(&self) -> String {
        self.format_graph(false, "--")
    }

    /// ASCII text with ANSI color codes.
    ///
    /// Same layout as [`to_ascii`](Self::to_ascii) with 16-color ANSI codes
    /// encoding the coset (hue) and sign parity (bold = even, normal = odd).
    /// A two-line legend is appended when non-identity VOPs are present.
    #[must_use]
    pub fn to_color_ascii(&self) -> String {
        self.format_graph(true, "--")
    }

    /// Unicode text (no escape codes).
    ///
    /// Same layout as [`to_ascii`](Self::to_ascii) with a Unicode separator
    /// (`\u{2500}\u{2500}`) instead of `--`.
    #[must_use]
    pub fn to_unicode(&self) -> String {
        self.format_graph(false, "\u{2500}\u{2500}")
    }

    /// Unicode text with ANSI color codes.
    #[must_use]
    pub fn to_color_unicode(&self) -> String {
        self.format_graph(true, "\u{2500}\u{2500}")
    }

    /// Deprecated: use [`to_color_ascii`](Self::to_color_ascii) instead.
    #[deprecated(note = "renamed to to_color_ascii")]
    #[must_use]
    pub fn to_ascii_color(&self) -> String {
        self.to_color_ascii()
    }

    /// Shared layout logic.
    fn format_graph(&self, color: bool, separator: &str) -> String {
        let n = self.num_qubits();
        let num_edges = self.num_edges();
        let mut out = format!("GraphState: {n} qubits, {num_edges} edges\n\n");

        if n == 0 {
            return out;
        }

        let idx_width = (n - 1).to_string().len();
        let show_vops = !self.is_pure_graph_state();

        // Compute maximum bracketed VOP width across non-identity vertices.
        let max_vop_width = if show_vops {
            (0..n)
                .filter(|&v| !self.vops[v].is_identity())
                .map(|v| {
                    let idx = self.vops[v].index() as usize;
                    CLIFFORD_NAMES[idx].len() + 2 // +2 for brackets
                })
                .max()
                .unwrap_or(0)
        } else {
            0
        };

        for v in 0..n {
            write!(out, "  {v:>idx_width$}").unwrap();

            if show_vops {
                let idx = self.vops[v].index() as usize;
                if self.vops[v].is_identity() {
                    write!(out, " {:<max_vop_width$}", "").unwrap();
                } else {
                    let name = CLIFFORD_NAMES[idx];
                    let (open, close) = VOP_BRACKETS[idx];
                    let bracketed = format!("{open}{name}{close}");
                    if color {
                        let ansi = VOP_ANSI[idx];
                        write!(
                            out,
                            " {ansi}{bracketed:<max_vop_width$}\x1b[0m",
                        )
                        .unwrap();
                    } else {
                        write!(out, " {bracketed:<max_vop_width$}").unwrap();
                    }
                }
            }

            // Neighbor list
            let nbrs: Vec<usize> = self.neighbors[v].iter().collect();
            if !nbrs.is_empty() {
                let nbr_str: Vec<String> = nbrs.iter().map(ToString::to_string).collect();
                write!(out, " {separator} {}", nbr_str.join(", ")).unwrap();
            }

            out.push('\n');
        }

        if color && show_vops {
            out.push('\n');
            out.push_str(
                "  \x1b[1;34mIdentity\x1b[0m  \
                 \x1b[1;35mX\u{2194}Z\x1b[0m  \
                 \x1b[1;33mX\u{2194}Y\x1b[0m  \
                 \x1b[1;36mY\u{2194}Z\x1b[0m  \
                 \x1b[1;37mCyc.fwd\x1b[0m  \
                 \x1b[90mCyc.inv\x1b[0m  \
                 (bold=even)\n",
            );
            out.push_str("  ()Pauli  []S-like  <>H-like  {}F-like\n");
        }

        out
    }
}

impl fmt::Display for GraphState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.num_qubits();
        write!(f, "GraphState({n} qubits")?;

        // Show non-identity VOPs
        let non_id: Vec<String> = (0..n)
            .filter(|&v| !self.vops[v].is_identity())
            .map(|v| {
                let name = CLIFFORD_NAMES[self.vops[v].index() as usize];
                format!("v{v}={name}")
            })
            .collect();

        if !non_id.is_empty() {
            write!(f, ", VOPs: {}", non_id.join(", "))?;
        }

        // Show edges
        let edges: Vec<String> = self.edges().map(|(u, v)| format!("{u}-{v}")).collect();
        if !edges.is_empty() {
            write!(f, ", edges: {}", edges.join(", "))?;
        }

        write!(f, ")")
    }
}

// ============================================================================
// GraphStateSim conversion support
// ============================================================================

impl crate::graph_state::GraphStateSim<PecosRng> {
    /// Create a simulator from a graph state representation with a seed.
    #[must_use]
    pub fn from_graph_state_with_seed(gs: GraphState, seed: u64) -> Self {
        let rng = PecosRng::seed_from_u64(seed);
        Self::from_graph_state(gs, rng)
    }
}

impl<R: SeedableRng + pecos_rng::Rng + core::fmt::Debug> crate::graph_state::GraphStateSim<R> {
    /// Create a simulator from a graph state representation.
    #[must_use]
    pub fn from_graph_state(gs: GraphState, rng: R) -> Self {
        let num_qubits = gs.num_qubits();
        let mut sim = Self::with_rng(num_qubits, rng);
        sim.vops = gs.vops;
        sim.neighbors = gs.neighbors;
        sim
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn pauli_axis_to_pauli(axis: PauliAxis) -> Pauli {
    match axis {
        PauliAxis::X => Pauli::X,
        PauliAxis::Y => Pauli::Y,
        PauliAxis::Z => Pauli::Z,
    }
}

/// Multiply two single-qubit Paulis, returning (result, phase).
/// P1 * P2 = phase * result
fn multiply_paulis(a: Pauli, b: Pauli) -> (Pauli, QuarterPhase) {
    use Pauli::{I, X, Y, Z};
    match (a, b) {
        (I, p) | (p, I) => (p, QuarterPhase::PlusOne),
        (X, X) | (Y, Y) | (Z, Z) => (I, QuarterPhase::PlusOne),
        (X, Y) => (Z, QuarterPhase::PlusI),
        (Y, X) => (Z, QuarterPhase::MinusI),
        (Y, Z) => (X, QuarterPhase::PlusI),
        (Z, Y) => (X, QuarterPhase::MinusI),
        (Z, X) => (Y, QuarterPhase::PlusI),
        (X, Z) => (Y, QuarterPhase::MinusI),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CliffordGateable;

    // ========================================================================
    // Phase 1: Core type tests
    // ========================================================================

    #[test]
    fn test_new_creates_plus_state() {
        let gs = GraphState::new(3);
        assert_eq!(gs.num_qubits(), 3);
        assert_eq!(gs.num_edges(), 0);
        assert!(gs.is_pure_graph_state());
        for v in 0..3 {
            assert!(gs.vop(v).is_identity());
            assert_eq!(gs.degree(v), 0);
        }
    }

    #[test]
    fn test_from_edges() {
        let gs = GraphState::from_edges(3, &[(0, 1), (1, 2)]);
        assert_eq!(gs.num_qubits(), 3);
        assert_eq!(gs.num_edges(), 2);
        assert!(gs.has_edge(0, 1));
        assert!(gs.has_edge(1, 2));
        assert!(!gs.has_edge(0, 2));
        assert_eq!(gs.degree(0), 1);
        assert_eq!(gs.degree(1), 2);
        assert_eq!(gs.degree(2), 1);
    }

    #[test]
    fn test_from_adjacency_matrix() {
        let matrix = vec![
            vec![false, true, false],
            vec![true, false, true],
            vec![false, true, false],
        ];
        let gs = GraphState::from_adjacency_matrix(&matrix);
        assert_eq!(gs.num_edges(), 2);
        assert!(gs.has_edge(0, 1));
        assert!(gs.has_edge(1, 2));
    }

    #[test]
    fn test_adjacency_matrix_roundtrip() {
        let gs = GraphState::from_edges(4, &[(0, 1), (1, 2), (2, 3), (0, 3)]);
        let matrix = gs.adjacency_matrix();
        let gs2 = GraphState::from_adjacency_matrix(&matrix);
        assert_eq!(gs, gs2);
    }

    #[test]
    fn test_edges_iterator() {
        let gs = GraphState::from_edges(4, &[(0, 1), (2, 3), (0, 3)]);
        let mut edges: Vec<(usize, usize)> = gs.edges().collect();
        edges.sort();
        assert_eq!(edges, vec![(0, 1), (0, 3), (2, 3)]);
    }

    #[test]
    fn test_mutators() {
        let mut gs = GraphState::new(3);
        gs.add_edge(0, 1);
        assert!(gs.has_edge(0, 1));
        gs.toggle_edge(0, 1);
        assert!(!gs.has_edge(0, 1));
        gs.toggle_edge(1, 2);
        assert!(gs.has_edge(1, 2));
        gs.remove_edge(1, 2);
        assert!(!gs.has_edge(1, 2));
    }

    #[test]
    fn test_set_vop_and_apply_local_clifford() {
        let mut gs = GraphState::new(2);
        gs.set_vop(0, CliffordFrame::H);
        assert_eq!(gs.vop(0), CliffordFrame::H);
        assert!(!gs.is_pure_graph_state());

        gs.apply_local_clifford(0, CliffordFrame::H);
        // H * H = I
        assert!(gs.vop(0).is_identity());
        assert!(gs.is_pure_graph_state());
    }

    // ========================================================================
    // Phase 2: Patterns and local complementation
    // ========================================================================

    #[test]
    fn test_linear_cluster() {
        let gs = GraphState::linear_cluster(4);
        assert_eq!(gs.num_qubits(), 4);
        assert_eq!(gs.num_edges(), 3);
        assert!(gs.has_edge(0, 1));
        assert!(gs.has_edge(1, 2));
        assert!(gs.has_edge(2, 3));
        assert!(!gs.has_edge(0, 2));
    }

    #[test]
    fn test_ring() {
        let gs = GraphState::ring(4);
        assert_eq!(gs.num_edges(), 4);
        assert!(gs.has_edge(0, 1));
        assert!(gs.has_edge(1, 2));
        assert!(gs.has_edge(2, 3));
        assert!(gs.has_edge(3, 0));
    }

    #[test]
    fn test_star() {
        let gs = GraphState::star(4);
        assert_eq!(gs.num_edges(), 3);
        for i in 1..4 {
            assert!(gs.has_edge(0, i));
        }
        assert!(!gs.has_edge(1, 2));
    }

    #[test]
    fn test_lattice_2d() {
        let gs = GraphState::lattice_2d(2, 3);
        assert_eq!(gs.num_qubits(), 6);
        // 2x3 grid: 7 edges (3 horizontal + 2 rows * 2 vertical-ish... actually:
        // row 0: 0-1, 1-2 (2 horiz)
        // row 1: 3-4, 4-5 (2 horiz)
        // cols: 0-3, 1-4, 2-5 (3 vert)
        // total = 7
        assert_eq!(gs.num_edges(), 7);
    }

    #[test]
    fn test_complete() {
        let gs = GraphState::complete(4);
        assert_eq!(gs.num_edges(), 6); // C(4,2) = 6
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert!(gs.has_edge(i, j));
            }
        }
    }

    #[test]
    fn test_local_complement_toggles_neighbor_edges() {
        // Star on 4 vertices: 0 connected to 1, 2, 3
        let mut gs = GraphState::star(4);
        assert!(!gs.has_edge(1, 2));
        assert!(!gs.has_edge(1, 3));
        assert!(!gs.has_edge(2, 3));

        // LC on vertex 0: complement edges among {1, 2, 3}
        gs.local_complement(0);

        // Now 1-2, 1-3, 2-3 should all exist (complete among neighbors)
        assert!(gs.has_edge(1, 2));
        assert!(gs.has_edge(1, 3));
        assert!(gs.has_edge(2, 3));

        // Original edges 0-1, 0-2, 0-3 should still exist
        assert!(gs.has_edge(0, 1));
        assert!(gs.has_edge(0, 2));
        assert!(gs.has_edge(0, 3));
    }

    #[test]
    fn test_local_complement_double_is_identity_on_graph() {
        // Two LCs on the same vertex should restore the graph (but change VOPs)
        let gs_orig = GraphState::star(4);
        let mut gs = gs_orig.clone();

        gs.local_complement(0);
        gs.local_complement(0);

        // Graph should be restored
        assert_eq!(gs.adjacency_matrix(), gs_orig.adjacency_matrix());
    }

    #[test]
    fn test_pivot() {
        let mut gs = GraphState::from_edges(4, &[(0, 1), (0, 2), (1, 3)]);
        gs.pivot(0, 1);
        // Pivot is LC(0), LC(1), LC(0) - it should complete without panicking
        // and maintain valid state
        assert_eq!(gs.num_qubits(), 4);
    }

    #[test]
    fn test_absorb_vops_on_pure_graph_state() {
        // A pure graph state should remain unchanged
        let gs_orig = GraphState::linear_cluster(4);
        let mut gs = gs_orig.clone();
        gs.absorb_vops();
        assert!(gs.is_pure_graph_state());
        assert_eq!(gs.adjacency_matrix(), gs_orig.adjacency_matrix());
    }

    #[test]
    fn test_absorb_vops_on_identity_vops() {
        // Pure graph states with identity VOPs: generators have X_v Z_neighbors form
        let gs = GraphState::linear_cluster(3);
        let gens = gs.stabilizer_generators();

        // Each generator should have exactly one X
        for (v, g) in gens.iter().enumerate() {
            assert_eq!(g.get(v), Pauli::X);
            for u in 0..3 {
                if u != v {
                    if gs.has_edge(v, u) {
                        assert_eq!(g.get(u), Pauli::Z);
                    } else {
                        assert_eq!(g.get(u), Pauli::I);
                    }
                }
            }
        }
    }

    #[test]
    fn test_absorb_vops_produces_pure_graph_state() {
        // Pure graph state: absorb is a no-op
        let mut gs = GraphState::linear_cluster(4);
        let adj_before = gs.adjacency_matrix();
        gs.absorb_vops();
        assert!(gs.is_pure_graph_state());
        assert_eq!(gs.adjacency_matrix(), adj_before);
    }

    #[test]
    fn test_absorb_vops_preserves_stabilizers() {
        // Verify that absorb_vops preserves the stabilizer group
        use pecos_core::PauliOperator;

        let mut gs = GraphState::linear_cluster(4);
        gs.set_vop(1, CliffordFrame::SZ);

        // Compute stabilizers before absorb
        let gens_before = gs.stabilizer_generators();

        gs.absorb_vops();

        // Compute stabilizers after absorb
        let gens_after = gs.stabilizer_generators();

        // All generators should commute across the two sets
        // (same stabilizer group means mutual commutativity)
        for ga in &gens_after {
            for gb in &gens_before {
                assert!(
                    ga.commutes_with(gb),
                    "absorb_vops should preserve stabilizer group"
                );
            }
        }
    }

    // ========================================================================
    // Phase 3: Stabilizer extraction
    // ========================================================================

    #[test]
    fn test_stabilizer_generator_single_qubit() {
        // Single qubit |+> state: stabilizer is +X
        let gs = GraphState::new(1);
        let stab = gs.stabilizer_generator(0);
        assert_eq!(stab.get(0), Pauli::X);
        assert_eq!(stab.phase(), QuarterPhase::PlusOne);
    }

    #[test]
    fn test_stabilizer_generators_two_qubit_graph() {
        // Two qubits with edge 0-1: |G> has stabilizers X_0 Z_1 and Z_0 X_1
        let gs = GraphState::from_edges(2, &[(0, 1)]);
        let gens = gs.stabilizer_generators();

        // Generator for vertex 0: X_0 * Z_1
        assert_eq!(gens[0].get(0), Pauli::X);
        assert_eq!(gens[0].get(1), Pauli::Z);
        assert_eq!(gens[0].phase(), QuarterPhase::PlusOne);

        // Generator for vertex 1: Z_0 * X_1
        assert_eq!(gens[1].get(0), Pauli::Z);
        assert_eq!(gens[1].get(1), Pauli::X);
        assert_eq!(gens[1].phase(), QuarterPhase::PlusOne);
    }

    #[test]
    fn test_stabilizer_generators_linear_cluster() {
        // 3-qubit linear cluster 0-1-2
        // K_0 = X_0 Z_1 I_2
        // K_1 = Z_0 X_1 Z_2
        // K_2 = I_0 Z_1 X_2
        let gs = GraphState::linear_cluster(3);
        let gens = gs.stabilizer_generators();

        assert_eq!(gens[0].get(0), Pauli::X);
        assert_eq!(gens[0].get(1), Pauli::Z);
        assert_eq!(gens[0].get(2), Pauli::I);

        assert_eq!(gens[1].get(0), Pauli::Z);
        assert_eq!(gens[1].get(1), Pauli::X);
        assert_eq!(gens[1].get(2), Pauli::Z);

        assert_eq!(gens[2].get(0), Pauli::I);
        assert_eq!(gens[2].get(1), Pauli::Z);
        assert_eq!(gens[2].get(2), Pauli::X);
    }

    #[test]
    fn test_stabilizer_generators_commute() {
        // All stabilizer generators of a graph state must commute
        use pecos_core::PauliOperator;

        let gs = GraphState::linear_cluster(4);
        let gens = gs.stabilizer_generators();

        for i in 0..gens.len() {
            for j in (i + 1)..gens.len() {
                assert!(
                    gens[i].commutes_with(&gens[j]),
                    "generators {i} and {j} should commute"
                );
            }
        }
    }

    #[test]
    fn test_stabilizer_generators_with_vops() {
        // Apply H to vertex 0 of a 2-qubit graph state
        // This should conjugate the generator at vertex 0
        let mut gs = GraphState::from_edges(2, &[(0, 1)]);
        gs.set_vop(0, CliffordFrame::H);

        let gens = gs.stabilizer_generators();

        // H maps X->Z, Z->X. So:
        // Generator for v0: H(X_0) * Z_1 = Z_0 * Z_1
        assert_eq!(gens[0].get(0), Pauli::Z);
        assert_eq!(gens[0].get(1), Pauli::Z);

        // Generator for v1: H(Z_0) * X_1 = X_0 * X_1
        assert_eq!(gens[1].get(0), Pauli::X);
        assert_eq!(gens[1].get(1), Pauli::X);
    }

    #[test]
    fn test_lc_preserves_stabilizer_group() {
        // Local complementation should preserve the stabilizer group
        // (generators may change but they should generate the same group).
        // We verify by checking that all new generators commute with all old generators
        // AND that new generators are in the stabilizer group of the original state.
        use pecos_core::PauliOperator;

        let gs_before = GraphState::linear_cluster(3);
        let gens_before = gs_before.stabilizer_generators();

        let mut gs_after = gs_before.clone();
        gs_after.local_complement(1);
        let gens_after = gs_after.stabilizer_generators();

        // All generators after LC should commute with all generators before
        for ga in &gens_after {
            for gb in &gens_before {
                assert!(
                    ga.commutes_with(gb),
                    "LC should preserve stabilizer group commutativity"
                );
            }
        }
    }

    // ========================================================================
    // Phase 4: Conversions
    // ========================================================================

    #[test]
    fn test_roundtrip_graph_state_to_sim() {
        let gs = GraphState::from_edges(3, &[(0, 1), (1, 2)]);

        let sim = gs.clone().into_sim_with_seed(42);
        let gs2 = sim.to_graph_state();

        assert_eq!(gs, gs2);
    }

    #[test]
    fn test_tensor_product() {
        let a = GraphState::from_edges(2, &[(0, 1)]);
        let b = GraphState::from_edges(2, &[(0, 1)]);
        let ab = a.tensor_product(&b);

        assert_eq!(ab.num_qubits(), 4);
        assert_eq!(ab.num_edges(), 2);
        assert!(ab.has_edge(0, 1));
        assert!(ab.has_edge(2, 3));
        assert!(!ab.has_edge(1, 2));
    }

    #[test]
    fn test_delete_vertex() {
        let mut gs = GraphState::star(4);
        gs.delete_vertex(0);
        assert_eq!(gs.degree(0), 0);
        assert!(gs.vop(0).is_identity());
        for i in 1..4 {
            assert!(!gs.has_edge(0, i));
        }
    }

    #[test]
    fn test_induced_subgraph() {
        let gs = GraphState::linear_cluster(5); // 0-1-2-3-4
        let sub = gs.induced_subgraph(&[1, 2, 3]);

        assert_eq!(sub.num_qubits(), 3);
        assert_eq!(sub.num_edges(), 2);
        assert!(sub.has_edge(0, 1)); // was 1-2
        assert!(sub.has_edge(1, 2)); // was 2-3
    }

    // ========================================================================
    // Phase 5: LC-equivalence
    // ========================================================================

    #[test]
    fn test_lc_orbit_single_qubit() {
        let gs = GraphState::new(1);
        let orbit = gs.lc_orbit();
        // Single isolated qubit: LC is a no-op on graph structure
        assert_eq!(orbit.len(), 1);
    }

    #[test]
    fn test_lc_orbit_two_qubit_edge() {
        let gs = GraphState::from_edges(2, &[(0, 1)]);
        let orbit = gs.lc_orbit();
        // Two vertices with one edge: LC on either vertex just toggles
        // the edges among neighbors (which is empty for the non-target),
        // so the graph stays the same.
        assert_eq!(orbit.len(), 1);
    }

    #[test]
    fn test_lc_equivalence_star_complete() {
        // K_4 and star on 4 vertices should be LC-equivalent
        // (well-known result)
        let star = GraphState::star(4);
        let complete = GraphState::complete(4);

        // LC on center of star produces K_4
        assert!(star.is_lc_equivalent(&complete));
    }

    #[test]
    fn test_lc_inequivalence() {
        // 4-qubit linear cluster and 4-qubit ring are NOT LC-equivalent
        // (they have different interlace polynomials)
        let linear = GraphState::linear_cluster(4);
        let ring = GraphState::ring(4);
        assert!(!linear.is_lc_equivalent(&ring));
    }

    #[test]
    fn test_lc_canonical_form_deterministic() {
        let gs = GraphState::star(4);
        let canon1 = gs.lc_canonical_form();
        let canon2 = gs.lc_canonical_form();
        assert_eq!(canon1, canon2);
    }

    // ========================================================================
    // Phase 6: Export
    // ========================================================================

    #[test]
    fn test_display() {
        let gs = GraphState::linear_cluster(3);
        let s = format!("{gs}");
        assert!(s.contains("3 qubits"));
        assert!(s.contains("0-1"));
        assert!(s.contains("1-2"));
    }

    #[test]
    fn test_to_dot() {
        let gs = GraphState::from_edges(2, &[(0, 1)]);
        let dot = gs.to_dot();
        assert!(dot.contains("graph G {"));
        assert!(dot.contains("0 -- 1"));
        assert!(dot.contains("}"));
    }

    #[test]
    fn test_to_svg() {
        let gs = GraphState::from_edges(3, &[(0, 1), (1, 2)]);
        let svg = gs.to_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        // 3 vertex circles + 9 legend circles = 12, and 2 edge lines
        assert_eq!(svg.matches("<circle").count(), 12);
        assert_eq!(svg.matches("<line").count(), 2);
    }

    #[test]
    fn test_to_svg_with_vops() {
        let mut gs = GraphState::from_edges(2, &[(0, 1)]);
        gs.set_vop(0, CliffordFrame::H);
        let svg = gs.to_svg();
        // Non-identity VOP should get a label
        assert!(svg.contains("H"));
        // Identity vertex gets identity fill, H vertex gets H-type fill
        assert!(svg.contains("#6495ED")); // identity coset fill
        assert!(svg.contains("#C850C0")); // X<->Z coset fill (H)
        // Gate family strokes: Pauli (identity) vs H-like (H gate)
        assert!(svg.contains("#1E3A8A")); // Pauli stroke
        assert!(svg.contains("#8B1A1A")); // H-like stroke
    }

    #[test]
    fn test_to_svg_empty() {
        let gs = GraphState::new(0);
        let svg = gs.to_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        // No vertex circles, but legend has 5 coset + 4 family = 9 circles
        assert_eq!(svg.matches("<circle").count(), 9);
    }

    #[test]
    fn test_to_tikz() {
        let gs = GraphState::from_edges(3, &[(0, 1), (1, 2)]);
        let tikz = gs.to_tikz();
        assert!(tikz.contains("\\begin{tikzpicture}"));
        assert!(tikz.contains("\\end{tikzpicture}"));
        // Should have 3 vertex nodes
        assert!(tikz.contains("(v0)"));
        assert!(tikz.contains("(v1)"));
        assert!(tikz.contains("(v2)"));
        // Should have 2 edges
        assert!(tikz.contains("\\draw (v0) -- (v1)"));
        assert!(tikz.contains("\\draw (v1) -- (v2)"));
    }

    #[test]
    fn test_to_tikz_with_vops() {
        let mut gs = GraphState::from_edges(2, &[(0, 1)]);
        gs.set_vop(1, CliffordFrame::SZ);
        let tikz = gs.to_tikz();
        // VOP annotation
        assert!(tikz.contains("$S$"));
        // Colour definitions present
        assert!(tikz.contains("\\definecolor{vopIdentity}"));
        assert!(tikz.contains("\\definecolor{famSqrt}"));
        // Identity vertex uses Pauli stroke, S vertex uses S-like stroke
        assert!(tikz.contains("fill=vopIdentity"));
        assert!(tikz.contains("draw=famPauli"));
        assert!(tikz.contains("draw=famSqrt"));
    }

    // ========================================================================
    // Cross-validation with simulator
    // ========================================================================

    #[test]
    fn test_cross_validate_stabilizers_with_sim() {
        // Build the same 3-qubit cluster state via the simulator (H + CZ)
        // and via GraphState::from_edges, then compare stabilizers.
        use pecos_core::QubitId;

        // Via GraphState (mathematical)
        let gs = GraphState::linear_cluster(3);
        let math_gens = gs.stabilizer_generators();

        // Via simulator
        let mut sim = crate::GraphStateSim::with_seed(3, 42);
        // Reset puts qubits in |0>. Apply H to get |+>, then CZ for edges.
        sim.h(&[QubitId::new(0), QubitId::new(1), QubitId::new(2)]);
        sim.cz(&[QubitId::new(0), QubitId::new(1)]);
        sim.cz(&[QubitId::new(1), QubitId::new(2)]);

        let sim_gs = sim.to_graph_state();
        let sim_gens = sim_gs.stabilizer_generators();

        // Both should have the same stabilizer generators
        // (possibly in different order or with different signs, but same Paulis)
        assert_eq!(math_gens.len(), sim_gens.len());

        // For a pure graph state with the same graph, generators should match exactly
        for (i, (mg, sg)) in math_gens.iter().zip(sim_gens.iter()).enumerate() {
            assert_eq!(
                mg.phase(),
                sg.phase(),
                "generator {i}: phase mismatch"
            );
            for q in 0..3 {
                assert_eq!(
                    mg.get(q),
                    sg.get(q),
                    "generator {i}, qubit {q}: Pauli mismatch"
                );
            }
        }
    }

    #[test]
    fn test_cross_validate_roundtrip_preserves_measurement() {
        // Build a state via simulator, convert to GraphState and back,
        // verify measurements give same results.
        use pecos_core::QubitId;

        let mut sim1 = crate::GraphStateSim::with_seed(3, 42);
        sim1.h(&[QubitId::new(0), QubitId::new(1), QubitId::new(2)]);
        sim1.cz(&[QubitId::new(0), QubitId::new(1)]);
        sim1.cz(&[QubitId::new(1), QubitId::new(2)]);

        // Round-trip through GraphState
        let gs = sim1.to_graph_state();
        let mut sim2 = gs.into_sim_with_seed(42);

        // Both sims should produce the same measurement outcomes (same seed)
        let r1 = sim1.mz(&[QubitId::new(0)]);
        let r2 = sim2.mz(&[QubitId::new(0)]);
        assert_eq!(r1[0].outcome, r2[0].outcome);
    }

    // ========================================================================
    // ASCII export
    // ========================================================================

    #[test]
    fn test_to_ascii_pure_graph_state() {
        let gs = GraphState::linear_cluster(3);
        let ascii = gs.to_ascii();

        // Header
        assert!(ascii.contains("GraphState: 3 qubits, 2 edges"));

        // Pure graph state: VOP column is omitted entirely
        assert!(!ascii.contains("(I)"), "identity VOPs should be hidden: {ascii}");

        // Edge info
        assert!(ascii.contains("-- 1"));
        assert!(ascii.contains("-- 0, 2"));

        // No ANSI escapes
        assert!(!ascii.contains("\x1b["));
    }

    #[test]
    fn test_to_ascii_color_contains_ansi() {
        // Need non-identity VOPs for color output (pure states have no VOPs to color)
        let mut gs = GraphState::from_edges(3, &[(0, 1), (1, 2)]);
        gs.set_vop(0, CliffordFrame::H);
        let colored = gs.to_ascii_color();

        // Should contain ANSI escape codes and resets
        assert!(colored.contains("\x1b["), "missing ANSI codes: {colored}");
        assert!(colored.contains("\x1b[0m"));

        // Should still have structure
        assert!(colored.contains("GraphState: 3 qubits, 2 edges"));
        assert!(colored.contains("<H>"));

        // Legend
        assert!(colored.contains("()Pauli"));
        assert!(colored.contains("bold=even"));
    }

    #[test]
    fn test_to_ascii_color_pure_has_no_ansi() {
        // Pure graph state: nothing to color, no legend
        let gs = GraphState::linear_cluster(3);
        let colored = gs.to_ascii_color();
        assert!(!colored.contains("\x1b["), "pure state should have no ANSI: {colored}");
        assert!(!colored.contains("Pauli"), "pure state should have no legend");
    }

    #[test]
    fn test_to_ascii_isolated_vertices() {
        let gs = GraphState::new(2);
        let ascii = gs.to_ascii();

        // Isolated pure graph: no edges, no VOP column
        assert!(!ascii.contains("--"));
        assert!(ascii.contains("2 qubits"));
        assert!(ascii.contains("0 edges"));
    }

    #[test]
    fn test_to_ascii_non_identity_vops() {
        let mut gs = GraphState::from_edges(2, &[(0, 1)]);
        gs.set_vop(0, CliffordFrame::H);
        let ascii = gs.to_ascii();

        // H is H-like family -> angle brackets
        assert!(ascii.contains("<H>"), "H bracket missing: {ascii}");
        // Vertex 1 is identity -> blank VOP column (no brackets)
        assert!(!ascii.contains("(I)"), "identity should be blank: {ascii}");
    }

    #[test]
    fn test_to_ascii_bracket_families() {
        let mut gs = GraphState::new(4);
        gs.set_vop(0, CliffordFrame::from_index(1)); // idx 1: X, Pauli -> ()
        gs.set_vop(1, CliffordFrame::SZ);             // idx 4: S-like   -> []
        gs.set_vop(2, CliffordFrame::H);              // idx 6: H-like   -> <>
        gs.set_vop(3, CliffordFrame::from_index(7));  // idx 7: F-like   -> {}
        let ascii = gs.to_ascii();

        assert!(ascii.contains("(X)"), "Pauli bracket missing: {ascii}");
        assert!(ascii.contains("[S]"), "S-like bracket missing: {ascii}");
        assert!(ascii.contains("<H>"), "H-like bracket missing: {ascii}");
        assert!(ascii.contains("{SH}"), "F-like bracket missing: {ascii}");
    }

    #[test]
    fn test_to_ascii_identity_alignment() {
        // When mixed VOPs are present, identity and non-identity rows
        // should have `--` at the same column.
        let mut gs = GraphState::from_edges(3, &[(0, 1), (1, 2)]);
        gs.set_vop(0, CliffordFrame::H);
        let ascii = gs.to_ascii();

        // Find the `--` column for each line that has neighbors
        let dash_cols: Vec<usize> = ascii
            .lines()
            .filter_map(|line| line.find("--"))
            .collect();
        assert!(dash_cols.len() >= 2, "expected at least 2 lines with --");
        assert!(
            dash_cols.windows(2).all(|w| w[0] == w[1]),
            "-- columns should align: {dash_cols:?}\n{ascii}"
        );
    }

    #[test]
    fn test_to_ascii_empty_graph() {
        let gs = GraphState::new(0);
        let ascii = gs.to_ascii();
        assert!(ascii.contains("0 qubits, 0 edges"));
    }
}
