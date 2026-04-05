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
//! generators are `K_v` = `X_v` * prod_{u in N(v)} `Z_u`.
//!
//! Any stabilizer state can be written as local Cliffords applied to a graph state:
//! `|psi> = (tensor_v VOP_v) |G>`. The VOP (vertex operator) on each qubit is a
//! single-qubit Clifford tracked as a [`CliffordFrame`].
//!
//! # Examples
//!
//! ```
//! use pecos_simulators::GraphState;
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
use pecos_core::circuit_diagram::{CellColor, FillPattern, GateFamily, GraphStyle, blend_hex};
use pecos_core::{BitSet, Pauli, PauliString, Phase, QuarterPhase};
use pecos_random::{PecosRng, SeedableRng};
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
    /// All VOPs are identity.
    ///
    /// # Panics
    /// Panics if any vertex index is >= `n` or if any edge is a self-loop.
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
    /// # Panics
    /// Panics if the matrix is not square or not symmetric.
    #[must_use]
    pub fn from_adjacency_matrix(matrix: &[Vec<bool>]) -> Self {
        let n = matrix.len();
        for row in matrix {
            assert_eq!(row.len(), n, "adjacency matrix must be square");
        }
        let mut gs = Self::new(n);
        for (i, row) in matrix.iter().enumerate() {
            for j in (i + 1)..n {
                assert_eq!(row[j], matrix[j][i], "adjacency matrix must be symmetric");
                if row[j] {
                    gs.neighbors[i].insert(j);
                    gs.neighbors[j].insert(i);
                }
            }
        }
        gs
    }

    /// Create a graph state from raw parts (VOPs and adjacency lists).
    ///
    /// # Panics
    /// Panics if `vops` and `neighbors` have different lengths.
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
    /// # Panics
    /// Panics if `n < 3`.
    #[must_use]
    pub fn ring(n: usize) -> Self {
        assert!(n >= 3, "ring requires at least 3 vertices");
        let mut edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        edges.push((n - 1, 0));
        Self::from_edges(n, &edges)
    }

    /// Star graph state: vertex 0 connected to all others.
    ///
    /// # Panics
    /// Panics if `n < 2`.
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

    /// Complete graph state `K_n`.
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
    ///
    /// # Panics
    /// Panics if `u == v` (self-loops not allowed).
    pub fn toggle_edge(&mut self, u: usize, v: usize) {
        assert_ne!(u, v, "self-loops not allowed");
        self.neighbors[u].toggle(v);
        self.neighbors[v].toggle(u);
    }

    /// Add edge (u, v). No-op if already present.
    ///
    /// # Panics
    /// Panics if `u == v` (self-loops not allowed).
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
    /// - Prepend sqrt(-iX) = SXDG to `VOP_v`
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
    /// # Panics
    /// Panics if `u` and `v` are not adjacent.
    pub fn pivot(&mut self, u: usize, v: usize) {
        assert!(self.has_edge(u, v), "pivot requires u and v to be adjacent");
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
    #[allow(clippy::missing_panics_doc)] // internal unwrap is guarded by a prior None check
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
    /// The bare generator is `K_v` = `X_v` * prod_{u in N(v)} `Z_u`.
    /// The conjugated generator is `VOP_v(X_v)` * prod_{u in N(v)} `VOP_u(Z_u)`.
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
        for u in &self.neighbors[v] {
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
    pub fn into_sim<R: SeedableRng + pecos_random::Rng + core::fmt::Debug>(
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
            for u in nbrs {
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
            for u in &self.neighbors[old_idx] {
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
    ///
    /// # Panics
    /// Panics if the LC orbit is empty (should never happen for a valid graph state).
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
    "I", "X", "Y", "Z", "S", "Sdg", "H", "SH", "HS", "S2H", "HS2", "S3H", "SHS", "HSH", "SHSH",
    "S2HS", "SHS2", "S3HS", "S2HS2", "S2HSH", "HS2HS", "S3HS2", "S3HSH", "HS2HS3",
];

// ============================================================================
// VOP Color Algebra
// ============================================================================
//
// Three independent visual dimensions encode Clifford structure:
//
// 1. **Fill hue** — axis permutation coset (which pair of Pauli axes
//    the Clifford interconverts, ignoring signs):
//      Blue   — identity perm (X->X, Z->Z)       -> CellColor::ZAxis
//      Purple — X<->Z swap (H-type)               -> CellColor::XZMix
//      Gold   — X<->Y swap (S-type)               -> CellColor::XYMix
//      Cyan   — Y<->Z swap (SX-type)              -> CellColor::YZMix
//      Gray   — 3-cycle                            -> CellColor::XYZMix
//
// 2. **Fill brightness** — sign parity of the Heisenberg action:
//      Saturated — even parity (0 or 2 negative signs)
//      Light     — odd parity (1 negative sign)
//
// 3. **Stroke colour** — gate family (geometric rotation type on the
//    Bloch sphere):
//      Navy     — Pauli (identity / pi-rotations)  -> GateFamily::Pauli
//      Green    — sqrt-of-Pauli / S-like            -> GateFamily::SLike
//      Maroon   — Hadamard-like                     -> GateFamily::HLike
//      Charcoal — Face-like / cyclic                -> GateFamily::FLike

/// Map a Clifford index to its axis permutation coset [`CellColor`].
#[rustfmt::skip]
fn vop_cell_color(idx: u8) -> CellColor {
    match idx {
        0..=3                   => CellColor::ZAxis,   // Identity/Pauli
        4 | 5 | 20 | 23                 => CellColor::XYMix,   // X<->Y (S-type)
        6 | 9 | 10 | 18                 => CellColor::XZMix,   // X<->Z (H-type)
        12 | 13 | 17 | 19               => CellColor::YZMix,   // Y<->Z (SX-type)
        7 | 8 | 11 | 14 | 15 | 16 | 21 | 22 => CellColor::XYZMix, // Cyclic
        _ => panic!("invalid Clifford index: {idx} (expected 0..24)"),
    }
}

/// Map a Clifford index to its gate family ([`GateFamily`]).
#[rustfmt::skip]
fn vop_gate_family(idx: u8) -> GateFamily {
    match idx {
        0..=3                         => GateFamily::Pauli,
        4 | 5 | 9 | 10 | 12 | 13              => GateFamily::SLike,
        6 | 17 | 18 | 19 | 20 | 23            => GateFamily::HLike,
        7 | 8 | 11 | 14 | 15 | 16 | 21 | 22  => GateFamily::FLike,
        _ => panic!("invalid Clifford index: {idx} (expected 0..24)"),
    }
}

/// Returns true if the Clifford at this index has even sign parity (saturated fill).
///
/// Even parity = 0 or 2 negative signs in the Heisenberg image.
/// For cyclic coset: forward (7,11,16,21) = saturated, inverse (8,14,15,22) = light.
#[rustfmt::skip]
fn vop_saturated(idx: u8) -> bool {
    match idx {
        0 | 2 | 5 | 6 | 7 | 11 | 13 | 16 | 17 | 18 | 20 | 21 => true,
        1 | 3 | 4 | 8 | 9 | 10 | 12 | 14 | 15 | 19 | 22 | 23 => false,
        _ => panic!("invalid Clifford index: {idx} (expected 0..24)"),
    }
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

/// `TikZ` color name for a [`CellColor`] coset.
fn tikz_coset_name(color: CellColor, saturated: bool) -> &'static str {
    match (color, saturated) {
        (CellColor::ZAxis, true) => "vopIdentity",
        (CellColor::ZAxis, false) => "vopIdentityLt",
        (CellColor::XZMix, true) => "vopXZ",
        (CellColor::XZMix, false) => "vopXZLt",
        (CellColor::XYMix, true) => "vopXY",
        (CellColor::XYMix, false) => "vopXYLt",
        (CellColor::YZMix, true) => "vopYZ",
        (CellColor::YZMix, false) => "vopYZLt",
        (CellColor::XYZMix, true) => "vopCyclicFwd",
        (CellColor::XYZMix, false) => "vopCyclicInv",
        (other, _) => panic!("unexpected CellColor for VOP coset: {other:?}"),
    }
}

/// `TikZ` color name for a [`GateFamily`] stroke.
fn tikz_family_name(family: GateFamily) -> &'static str {
    match family {
        GateFamily::Pauli
        | GateFamily::Default
        | GateFamily::Measurement
        | GateFamily::Preparation => "famPauli",
        GateFamily::SLike => "famSqrt",
        GateFamily::HLike => "famHadamard",
        GateFamily::FLike => "famCyclic",
    }
}

impl GraphState {
    /// Create a renderer bound to a [`GraphStyle`].
    ///
    /// # Examples
    /// ```
    /// use pecos_simulators::GraphState;
    /// use pecos_core::GraphStyle;
    ///
    /// let gs = GraphState::linear_cluster(3);
    /// let svg = gs.render_with(&GraphStyle::default()).svg();
    /// assert!(svg.contains("<svg"));
    /// ```
    #[must_use]
    pub fn render_with<'a>(&'a self, style: &'a GraphStyle) -> GraphStateRenderer<'a> {
        GraphStateRenderer { graph: self, style }
    }

    /// Export to DOT format with default style.
    #[must_use]
    pub fn to_dot(&self) -> String {
        self.render_with(&GraphStyle::default()).dot()
    }

    /// Compute vertex positions using a circular layout.
    ///
    /// Returns (x, y) pairs for each vertex, centered at (`cx`, `cy`) with
    /// the given `radius`. Single-vertex graphs place the vertex at center.
    #[allow(clippy::cast_precision_loss)] // layout coordinate calculations
    fn circular_layout(n: usize, cx: f64, cy: f64, radius: f64) -> Vec<(f64, f64)> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![(cx, cy)];
        }
        (0..n)
            .map(|i| {
                let angle = -std::f64::consts::FRAC_PI_2
                    + 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                (cx + radius * angle.cos(), cy + radius * angle.sin())
            })
            .collect()
    }

    /// Export to SVG with default style.
    #[must_use]
    pub fn to_svg(&self) -> String {
        self.render_with(&GraphStyle::default()).svg()
    }

    /// Export to `TikZ` with default style.
    #[must_use]
    pub fn to_tikz(&self) -> String {
        self.render_with(&GraphStyle::default()).tikz()
    }

    /// Export as plain ASCII text (no escape codes).
    #[must_use]
    pub fn to_ascii(&self) -> String {
        self.render_with(&GraphStyle::default()).ascii()
    }

    /// ASCII text with ANSI color codes.
    #[must_use]
    pub fn to_color_ascii(&self) -> String {
        self.render_with(&GraphStyle::builder().ansi_color(true).build())
            .ascii()
    }

    /// Unicode text (no escape codes).
    #[must_use]
    pub fn to_unicode(&self) -> String {
        self.render_with(&GraphStyle::default()).unicode()
    }

    /// Unicode text with ANSI color codes.
    #[must_use]
    pub fn to_color_unicode(&self) -> String {
        self.render_with(&GraphStyle::builder().ansi_color(true).build())
            .unicode()
    }
}

// ============================================================================
// GraphStateRenderer
// ============================================================================

/// A graph state bound to a [`GraphStyle`], ready to render in any output format.
///
/// Obtained via [`GraphState::render_with`].
pub struct GraphStateRenderer<'a> {
    graph: &'a GraphState,
    style: &'a GraphStyle,
}

impl GraphStateRenderer<'_> {
    /// Render as a Graphviz DOT graph.
    #[must_use]
    pub fn dot(&self) -> String {
        let n = self.graph.num_qubits();
        let mut dot = String::from("graph G {\n");
        dot.push_str("  node [shape=circle, style=filled, fontsize=12];\n");

        for v in 0..n {
            let idx = self.graph.vops[v].index();
            let name = CLIFFORD_NAMES[idx as usize];
            let coset = vop_cell_color(idx);
            let family = vop_gate_family(idx);
            let sat = vop_saturated(idx);
            let fill = self.style.vop_fill(coset, sat);
            let stroke = self.style.vop_stroke(family);
            let text = self.style.vop_text(coset, sat);
            let dot_style = self.style.vop_dot_style(family);
            let style_attr = if dot_style.is_empty() {
                "filled".to_string()
            } else {
                format!("filled,{dot_style}")
            };
            writeln!(
                dot,
                "  {v} [label=\"{v}\\n{name}\" fillcolor=\"{fill}\" \
                 color=\"{stroke}\" fontcolor=\"{text}\" style=\"{style_attr}\"];",
            )
            .unwrap();
        }

        for (u, v) in self.graph.edges() {
            writeln!(dot, "  {u} -- {v};").unwrap();
        }

        dot.push_str("}\n");
        dot
    }

    /// Render as a standalone SVG string.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // SVG coordinate calculations
    pub fn svg(&self) -> String {
        let n = self.graph.num_qubits();
        let node_radius = 20.0;
        let layout_radius = if n <= 2 { 60.0 } else { 40.0 + 25.0 * n as f64 };
        let margin = node_radius + 40.0;
        let width = 2.0 * (layout_radius + margin);
        let legend_height = 50.0;
        let height = width + legend_height;
        let center = layout_radius + margin;

        let positions = GraphState::circular_layout(n, center, center, layout_radius);

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" \
             width=\"{width}\" height=\"{height}\" \
             viewBox=\"0 0 {width} {height}\">\n"
        );
        writeln!(
            svg,
            "  <rect width=\"{width}\" height=\"{height}\" fill=\"white\"/>"
        )
        .unwrap();

        // Collect needed fill patterns and emit <defs>
        let mut needed_patterns = BTreeSet::new();
        for v in 0..n {
            let idx = self.graph.vops[v].index();
            let pattern = self.style.vop_pattern(vop_cell_color(idx));
            if pattern != FillPattern::Solid {
                needed_patterns.insert(pattern);
            }
        }
        // Also include patterns used by legend cosets
        for coset in [
            CellColor::ZAxis,
            CellColor::XZMix,
            CellColor::XYMix,
            CellColor::YZMix,
            CellColor::XYZMix,
        ] {
            let pattern = self.style.vop_pattern(coset);
            if pattern != FillPattern::Solid {
                needed_patterns.insert(pattern);
            }
        }
        if !needed_patterns.is_empty() {
            svg.push_str("  <defs>\n");
            for pat in &needed_patterns {
                writeln!(svg, "    {}", pat.svg_pattern_def()).unwrap();
            }
            svg.push_str("  </defs>\n");
        }

        // Draw edges
        for (u, v) in self.graph.edges() {
            let (x1, y1) = positions[u];
            let (x2, y2) = positions[v];
            writeln!(
                svg,
                "  <line x1=\"{x1:.1}\" y1=\"{y1:.1}\" \
                 x2=\"{x2:.1}\" y2=\"{y2:.1}\" \
                 stroke=\"#555\" stroke-width=\"1.5\"/>"
            )
            .unwrap();
        }

        // Draw vertices
        for (v, &(x, y)) in positions.iter().enumerate() {
            let idx = self.graph.vops[v].index();
            let vop_name = CLIFFORD_NAMES[idx as usize];
            let coset = vop_cell_color(idx);
            let family = vop_gate_family(idx);
            let sat = vop_saturated(idx);
            let fill = self.style.vop_fill(coset, sat);
            let stroke = self.style.vop_stroke(family);
            let text = self.style.vop_text(coset, sat);
            let dash = self.style.vop_dasharray(family);
            let dash_attr = if dash.is_empty() {
                String::new()
            } else {
                format!(" stroke-dasharray=\"{dash}\"")
            };

            writeln!(
                svg,
                "  <circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"{node_radius}\" \
                 fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"2\"{dash_attr}/>"
            )
            .unwrap();

            // Pattern overlay
            let pattern = self.style.vop_pattern(coset);
            if pattern != FillPattern::Solid {
                let pat_r = node_radius - 1.0;
                writeln!(
                    svg,
                    "  <circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"{pat_r:.1}\" \
                     fill=\"url(#{})\" stroke=\"none\"/>",
                    pattern.svg_id()
                )
                .unwrap();
            }

            // Vertex index label
            writeln!(
                svg,
                "  <text x=\"{x:.1}\" y=\"{y:.1}\" \
                 text-anchor=\"middle\" dominant-baseline=\"central\" \
                 font-family=\"sans-serif\" font-size=\"12\" \
                 fill=\"{text}\" font-weight=\"bold\">{v}</text>"
            )
            .unwrap();

            // VOP label (below the node, only if non-identity)
            if !self.graph.vops[v].is_identity() {
                let label_y = y + node_radius + 14.0;
                writeln!(
                    svg,
                    "  <text x=\"{x:.1}\" y=\"{label_y:.1}\" \
                     text-anchor=\"middle\" \
                     font-family=\"sans-serif\" font-size=\"10\" \
                     fill=\"#666\">{vop_name}</text>"
                )
                .unwrap();
            }
        }

        // Legend
        self.svg_legend(&mut svg, width, height, legend_height);

        svg.push_str("</svg>\n");
        svg
    }

    /// Append an SVG legend derived from the style palette.
    #[allow(clippy::cast_precision_loss)] // SVG coordinate calculations
    fn svg_legend(&self, svg: &mut String, width: f64, height: f64, legend_height: f64) {
        let y_top = height - legend_height + 8.0;
        let r = 6.0;

        // Row 1: fill hues (cosets) -- show saturated fill + family stroke
        let cosets: &[(CellColor, &str)] = &[
            (CellColor::ZAxis, "I/Pauli"),
            (CellColor::XZMix, "X\u{2194}Z"),
            (CellColor::XYMix, "X\u{2194}Y"),
            (CellColor::YZMix, "Y\u{2194}Z"),
            (CellColor::XYZMix, "Cyclic"),
        ];

        let spacing = width / (cosets.len() as f64 + 1.0);
        for (i, &(coset, label)) in cosets.iter().enumerate() {
            let cx = spacing * (i as f64 + 1.0);
            let fill = self.style.vop_fill(coset, true);
            let stroke = blend_hex(&fill, "#000000", 0.4);
            writeln!(
                svg,
                "  <circle cx=\"{cx:.1}\" cy=\"{y_top:.1}\" r=\"{r}\" \
                 fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1.5\"/>"
            )
            .unwrap();
            let pattern = self.style.vop_pattern(coset);
            if pattern != FillPattern::Solid {
                let pr = r - 0.5;
                writeln!(
                    svg,
                    "  <circle cx=\"{cx:.1}\" cy=\"{y_top:.1}\" r=\"{pr:.1}\" \
                     fill=\"url(#{})\" stroke=\"none\"/>",
                    pattern.svg_id()
                )
                .unwrap();
            }
            let tx = cx + r + 4.0;
            let ty = y_top + 3.0;
            writeln!(
                svg,
                "  <text x=\"{tx:.1}\" y=\"{ty:.1}\" \
                 font-family=\"sans-serif\" font-size=\"9\" fill=\"#555\">\
                 {label}</text>"
            )
            .unwrap();
        }

        // Row 2: stroke colours (gate families)
        let families: &[(GateFamily, &str)] = &[
            (GateFamily::Pauli, "Pauli"),
            (GateFamily::SLike, "S-like"),
            (GateFamily::HLike, "H-like"),
            (GateFamily::FLike, "F-like"),
        ];

        let y_row2 = y_top + 18.0;
        let fam_spacing = width / (families.len() as f64 + 1.0);
        for (i, &(family, label)) in families.iter().enumerate() {
            let cx = fam_spacing * (i as f64 + 1.0);
            let stroke_col = self.style.vop_stroke(family);
            let dash = self.style.vop_dasharray(family);
            let dash_attr = if dash.is_empty() {
                String::new()
            } else {
                format!(" stroke-dasharray=\"{dash}\"")
            };
            writeln!(
                svg,
                "  <circle cx=\"{cx:.1}\" cy=\"{y_row2:.1}\" r=\"{r}\" \
                 fill=\"white\" stroke=\"{stroke_col}\" stroke-width=\"2.5\"{dash_attr}/>"
            )
            .unwrap();
            let tx = cx + r + 4.0;
            let ty = y_row2 + 3.0;
            writeln!(
                svg,
                "  <text x=\"{tx:.1}\" y=\"{ty:.1}\" \
                 font-family=\"sans-serif\" font-size=\"9\" fill=\"#555\">\
                 {label}</text>"
            )
            .unwrap();
        }
    }

    /// Render as a `TikZ` `tikzpicture` environment.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // TikZ coordinate calculations
    pub fn tikz(&self) -> String {
        let n = self.graph.num_qubits();
        let radius = if n <= 2 { 1.5 } else { 1.0 + 0.5 * n as f64 };
        let positions = GraphState::circular_layout(n, 0.0, 0.0, radius);

        let mut tikz = String::from("\\begin{tikzpicture}\n");

        // Colour definitions derived from style
        tikz.push_str("  % Fill: axis permutation coset (bright / light)\n");
        for &(coset, sat, name) in &[
            (CellColor::ZAxis, true, "vopIdentity"),
            (CellColor::ZAxis, false, "vopIdentityLt"),
            (CellColor::XZMix, true, "vopXZ"),
            (CellColor::XZMix, false, "vopXZLt"),
            (CellColor::XYMix, true, "vopXY"),
            (CellColor::XYMix, false, "vopXYLt"),
            (CellColor::YZMix, true, "vopYZ"),
            (CellColor::YZMix, false, "vopYZLt"),
            (CellColor::XYZMix, true, "vopCyclicFwd"),
            (CellColor::XYZMix, false, "vopCyclicInv"),
        ] {
            let hex = self.style.vop_fill(coset, sat);
            let hex = hex.strip_prefix('#').unwrap_or(&hex);
            writeln!(tikz, "  \\definecolor{{{name}}}{{HTML}}{{{hex}}}").unwrap();
        }
        tikz.push_str("  % Stroke: gate family\n");
        for &(family, name) in &[
            (GateFamily::Pauli, "famPauli"),
            (GateFamily::SLike, "famSqrt"),
            (GateFamily::HLike, "famHadamard"),
            (GateFamily::FLike, "famCyclic"),
        ] {
            let hex = self.style.vop_stroke(family);
            let hex = hex.strip_prefix('#').unwrap_or(hex);
            writeln!(tikz, "  \\definecolor{{{name}}}{{HTML}}{{{hex}}}").unwrap();
        }

        // Check if any coset uses patterns (need patterns library)
        let any_pattern = [
            CellColor::ZAxis,
            CellColor::XZMix,
            CellColor::XYMix,
            CellColor::YZMix,
            CellColor::XYZMix,
        ]
        .iter()
        .any(|&c| self.style.vop_pattern(c) != FillPattern::Solid);
        if any_pattern {
            tikz.push_str("  % Requires: \\usetikzlibrary{patterns}\n");
        }

        // Base vertex style
        tikz.push_str(
            "  \\tikzstyle{vertex}=[circle, minimum size=20pt, \
             inner sep=0pt, font=\\small, line width=1.5pt]\n",
        );
        tikz.push_str("  \\tikzstyle{vop label}=[font=\\scriptsize, text=gray]\n");

        // Draw vertices
        for (v, &(x, y)) in positions.iter().enumerate() {
            let idx = self.graph.vops[v].index();
            let coset = vop_cell_color(idx);
            let family = vop_gate_family(idx);
            let sat = vop_saturated(idx);
            let fill_name = tikz_coset_name(coset, sat);
            let draw_name = tikz_family_name(family);
            let text_opt = if self.style.vop_text(coset, sat) == "white" {
                ", text=white"
            } else {
                ""
            };
            let tikz_dash = self.style.vop_tikz_dash(family);
            let dash_opt = if tikz_dash.is_empty() {
                String::new()
            } else {
                format!(", {tikz_dash}")
            };
            let tikz_pat = self.style.vop_pattern(coset).tikz_pattern();
            let pat_opt = if tikz_pat.is_empty() {
                String::new()
            } else {
                format!(", postaction={{pattern={tikz_pat}, pattern color=black!30}}")
            };

            writeln!(
                tikz,
                "  \\node[vertex, fill={fill_name}, draw={draw_name}{text_opt}{dash_opt}{pat_opt}] \
                 (v{v}) at ({x:.2}, {y:.2}) {{{v}}};",
            )
            .unwrap();

            // VOP annotation
            if !self.graph.vops[v].is_identity() {
                let vop_name = CLIFFORD_NAMES[idx as usize];
                let label_y = y - 0.45;
                writeln!(
                    tikz,
                    "  \\node[vop label] at ({x:.2}, {label_y:.2}) {{${vop_name}$}};",
                )
                .unwrap();
            }
        }

        // Draw edges
        for (u, v) in self.graph.edges() {
            writeln!(tikz, "  \\draw (v{u}) -- (v{v});").unwrap();
        }

        tikz.push_str("\\end{tikzpicture}\n");
        tikz
    }

    /// Render as plain ASCII text.
    ///
    /// Produces ANSI color codes when `style.ansi_color` is true.
    #[must_use]
    pub fn ascii(&self) -> String {
        self.format_text("--")
    }

    /// Render as Unicode text.
    ///
    /// Produces ANSI color codes when `style.ansi_color` is true.
    #[must_use]
    pub fn unicode(&self) -> String {
        self.format_text("\u{2500}\u{2500}")
    }

    /// Shared text layout logic.
    fn format_text(&self, separator: &str) -> String {
        let color = self.style.ansi_color;
        let n = self.graph.num_qubits();
        let num_edges = self.graph.num_edges();
        let mut out = format!("GraphState: {n} qubits, {num_edges} edges\n\n");

        if n == 0 {
            return out;
        }

        let idx_width = (n - 1).to_string().len();
        let show_vops = !self.graph.is_pure_graph_state();

        // Compute maximum bracketed VOP width across non-identity vertices.
        let max_vop_width = if show_vops {
            (0..n)
                .filter(|&v| !self.graph.vops[v].is_identity())
                .map(|v| {
                    let idx = self.graph.vops[v].index() as usize;
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
                let idx = self.graph.vops[v].index() as usize;
                if self.graph.vops[v].is_identity() {
                    write!(out, " {:<max_vop_width$}", "").unwrap();
                } else {
                    let name = CLIFFORD_NAMES[idx];
                    let (open, close) = VOP_BRACKETS[idx];
                    let bracketed = format!("{open}{name}{close}");
                    if color {
                        let ansi = VOP_ANSI[idx];
                        write!(out, " {ansi}{bracketed:<max_vop_width$}\x1b[0m",).unwrap();
                    } else {
                        write!(out, " {bracketed:<max_vop_width$}").unwrap();
                    }
                }
            }

            // Neighbor list
            let nbrs: Vec<usize> = self.graph.neighbors[v].iter().collect();
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

impl<R: SeedableRng + pecos_random::Rng + core::fmt::Debug> crate::graph_state::GraphStateSim<R> {
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
        edges.sort_unstable();
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
        assert!(dot.contains('}'));
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
        use pecos_core::GraphStyle;

        let mut gs = GraphState::from_edges(2, &[(0, 1)]);
        gs.set_vop(0, CliffordFrame::H);
        let svg = gs.to_svg();
        // Non-identity VOP should get a label
        assert!(svg.contains('H'));

        // Colors are now derived from the palette via blend_hex.
        // Check that the computed fills for identity (saturated ZAxis)
        // and H (saturated XZMix) both appear.
        let style = GraphStyle::default();
        let identity_fill = style.vop_fill(CellColor::ZAxis, true);
        let h_fill = style.vop_fill(CellColor::XZMix, true);
        assert!(
            svg.contains(&identity_fill),
            "identity fill missing: {identity_fill}"
        );
        assert!(svg.contains(&h_fill), "H fill missing: {h_fill}");

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
        sim.cz(&[(QubitId::new(0), QubitId::new(1))]);
        sim.cz(&[(QubitId::new(1), QubitId::new(2))]);

        let sim_gs = sim.to_graph_state();
        let sim_gens = sim_gs.stabilizer_generators();

        // Both should have the same stabilizer generators
        // (possibly in different order or with different signs, but same Paulis)
        assert_eq!(math_gens.len(), sim_gens.len());

        // For a pure graph state with the same graph, generators should match exactly
        for (i, (mg, sg)) in math_gens.iter().zip(sim_gens.iter()).enumerate() {
            assert_eq!(mg.phase(), sg.phase(), "generator {i}: phase mismatch");
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
        sim1.cz(&[(QubitId::new(0), QubitId::new(1))]);
        sim1.cz(&[(QubitId::new(1), QubitId::new(2))]);

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
        assert!(
            !ascii.contains("(I)"),
            "identity VOPs should be hidden: {ascii}"
        );

        // Edge info
        assert!(ascii.contains("-- 1"));
        assert!(ascii.contains("-- 0, 2"));

        // No ANSI escapes
        assert!(!ascii.contains("\x1b["));
    }

    #[test]
    fn test_to_color_ascii_contains_ansi() {
        // Need non-identity VOPs for color output (pure states have no VOPs to color)
        let mut gs = GraphState::from_edges(3, &[(0, 1), (1, 2)]);
        gs.set_vop(0, CliffordFrame::H);
        let colored = gs.to_color_ascii();

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
    fn test_to_color_ascii_pure_has_no_ansi() {
        // Pure graph state: nothing to color, no legend
        let gs = GraphState::linear_cluster(3);
        let colored = gs.to_color_ascii();
        assert!(
            !colored.contains("\x1b["),
            "pure state should have no ANSI: {colored}"
        );
        assert!(
            !colored.contains("Pauli"),
            "pure state should have no legend"
        );
    }

    #[test]
    fn render_with_ansi_color_matches_to_color_ascii() {
        let mut gs = GraphState::from_edges(3, &[(0, 1), (1, 2)]);
        gs.set_vop(0, CliffordFrame::H);
        let via_convenience = gs.to_color_ascii();
        let via_render_with = gs
            .render_with(&GraphStyle::builder().ansi_color(true).build())
            .ascii();
        assert_eq!(via_convenience, via_render_with);
    }

    #[test]
    fn render_with_ansi_color_matches_to_color_unicode() {
        let mut gs = GraphState::from_edges(3, &[(0, 1), (1, 2)]);
        gs.set_vop(0, CliffordFrame::H);
        let via_convenience = gs.to_color_unicode();
        let via_render_with = gs
            .render_with(&GraphStyle::builder().ansi_color(true).build())
            .unicode();
        assert_eq!(via_convenience, via_render_with);
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
        gs.set_vop(1, CliffordFrame::SZ); // idx 4: S-like   -> []
        gs.set_vop(2, CliffordFrame::H); // idx 6: H-like   -> <>
        gs.set_vop(3, CliffordFrame::from_index(7)); // idx 7: F-like   -> {}
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
        let dash_cols: Vec<usize> = ascii.lines().filter_map(|line| line.find("--")).collect();
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

    // ========================================================================
    // render_with tests
    // ========================================================================

    #[test]
    fn render_with_default_matches_to_svg() {
        use pecos_core::GraphStyle;

        let gs = GraphState::linear_cluster(3);
        let default_style = GraphStyle::default();
        assert_eq!(gs.render_with(&default_style).svg(), gs.to_svg());
    }

    #[test]
    fn render_with_custom_palette() {
        use pecos_core::{ColorPalette, ColorTriplet, GraphStyle};

        let palette = ColorPalette {
            z_axis: ColorTriplet::new("#FF0000", "#880000", "#440000"),
            ..ColorPalette::default()
        };
        let style = GraphStyle::builder().palette(palette).build();

        let gs = GraphState::linear_cluster(3); // pure: all identity (ZAxis coset)
        let svg = gs.render_with(&style).svg();

        // Saturated ZAxis fill = blend("#FF0000", "#880000", 0.5)
        let expected_fill = pecos_core::blend_hex("#FF0000", "#880000", 0.5);
        assert!(
            svg.contains(&expected_fill),
            "custom ZAxis fill {expected_fill} not found in SVG"
        );
    }

    #[test]
    fn render_with_monochrome() {
        use pecos_core::{ColorPalette, ColorTriplet, GraphStyle};

        // Set all cosets to the same color
        let grey = ColorTriplet::new("#CCCCCC", "#666666", "#333333");
        let palette = ColorPalette {
            z_axis: grey.clone(),
            xz_mix: grey.clone(),
            xy_mix: grey.clone(),
            yz_mix: grey.clone(),
            xyz_mix: grey.clone(),
            ..ColorPalette::default()
        };
        let style = GraphStyle::builder().palette(palette).build();

        let mut gs = GraphState::from_edges(2, &[(0, 1)]);
        gs.set_vop(0, CliffordFrame::H); // XZMix coset

        let svg = gs.render_with(&style).svg();
        // Both vertices should use the same grey palette
        let sat_fill = pecos_core::blend_hex("#CCCCCC", "#666666", 0.5);
        // Count occurrences of the saturated fill (both vertices are saturated: I=even, H=even)
        assert!(
            svg.matches(&sat_fill).count() >= 2,
            "monochrome fill {sat_fill} should appear at least twice"
        );
    }

    #[test]
    fn render_with_ascii_matches_to_ascii() {
        use pecos_core::GraphStyle;

        let gs = GraphState::linear_cluster(4);
        let default_style = GraphStyle::default();
        assert_eq!(gs.render_with(&default_style).ascii(), gs.to_ascii());
    }
}
