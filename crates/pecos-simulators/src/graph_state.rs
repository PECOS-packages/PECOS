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

//! Graph state stabilizer simulator inspired by the Anders & Briegel algorithm.
//!
//! Any stabilizer state can be written as local Cliffords applied to a graph state:
//! `|psi> = (tensor_v VOP_v) |G>` where `|G>` is the graph state defined by an
//! adjacency graph. Single-qubit Clifford gates are O(1) VOP updates. Two-qubit
//! gates and measurements require local complementation operations that are
//! O(degree) or O(degree^2).
//!
//! # References
//! - Anders & Briegel, "Fast simulation of stabilizer circuits using a graph-state
//!   representation", [quant-ph/0504117](https://arxiv.org/abs/quant-ph/0504117)

use crate::clifford_frame::{CliffordFrame, PauliAxis};
use crate::{CliffordGateable, MeasurementResult, QuantumSimulator};
use core::fmt::Debug;
use pecos_core::{BitSet, QubitId, RngManageable};
use pecos_random::rng_ext::RngProbabilityExt;
use pecos_random::{PecosRng, Rng, SeedableRng};

use crate::stabilizer_test_utils::{ForcedMeasurement, StabilizerSimulator};

/// Graph state stabilizer simulator.
///
/// Represents a stabilizer state as `|psi> = (tensor_v VOP_v) |G>` where
/// `VOP_v` is a single-qubit Clifford (vertex operator) on each qubit and
/// `|G>` is the graph state defined by the adjacency graph.
///
/// Single-qubit gates are O(1). Two-qubit gates are O(degree) amortized.
/// Measurements are O(degree^2) in the worst case.
#[derive(Clone, Debug)]
pub struct GraphStateSim<R: SeedableRng + Rng + Debug = PecosRng> {
    num_qubits: usize,
    /// Vertex operators: one single-qubit Clifford per qubit.
    pub(crate) vops: Vec<CliffordFrame>,
    /// Adjacency lists: `neighbors[v]` is the set of vertices adjacent to v.
    pub(crate) neighbors: Vec<BitSet>,
    rng: R,
}

// ============================================================================
// Constructors
// ============================================================================

impl GraphStateSim<PecosRng> {
    /// Create a new graph state simulator with the default RNG.
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let rng = rand::make_rng();
        Self::with_rng(num_qubits, rng)
    }

    /// Create a new graph state simulator with a specific seed.
    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let rng = PecosRng::seed_from_u64(seed);
        Self::with_rng(num_qubits, rng)
    }
}

impl<R: SeedableRng + Rng + Debug> GraphStateSim<R> {
    /// Create a new graph state simulator with a custom RNG.
    #[inline]
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        let mut state = Self {
            num_qubits,
            vops: vec![CliffordFrame::IDENTITY; num_qubits],
            neighbors: vec![BitSet::new(); num_qubits],
            rng,
        };
        state.reset();
        state
    }

    /// Returns the number of qubits.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Extract the graph state representation (cloning VOPs and neighbors).
    #[must_use]
    pub fn to_graph_state(&self) -> crate::graph_state_repr::GraphState {
        crate::graph_state_repr::GraphState::from_parts(self.vops.clone(), self.neighbors.clone())
    }

    /// Consume this simulator and return the graph state representation.
    #[must_use]
    pub fn into_graph_state(self) -> crate::graph_state_repr::GraphState {
        crate::graph_state_repr::GraphState::from_parts(self.vops, self.neighbors)
    }

    // ========================================================================
    // Internal: adjacency helpers
    // ========================================================================

    /// Toggle edge (a, b) in the graph.
    #[inline]
    fn toggle_edge(&mut self, a: usize, b: usize) {
        self.neighbors[a].toggle(b);
        self.neighbors[b].toggle(a);
    }

    /// Disconnect vertex a from all neighbors.
    fn disconnect(&mut self, a: usize) {
        // Collect neighbors to avoid borrow issues
        let nbrs: Vec<usize> = self.neighbors[a].iter().collect();
        for &u in &nbrs {
            self.neighbors[u].toggle(a);
        }
        self.neighbors[a].clear();
    }

    // ========================================================================
    // Internal: local complementation
    // ========================================================================

    /// Perform local complementation about vertex `a`.
    ///
    /// This complements all edges among neighbors of `a`, then updates VOPs:
    /// - Prepend sqrt(-iX) to `VOP_a`
    /// - Prepend sqrt(iZ) to each neighbor's VOP
    fn local_complement(&mut self, a: usize) {
        let nbrs: Vec<usize> = self.neighbors[a].iter().collect();

        // Complement edges among N(a)
        for i in 0..nbrs.len() {
            for j in (i + 1)..nbrs.len() {
                self.toggle_edge(nbrs[i], nbrs[j]);
            }
        }

        // Update VOPs: prepend sqrt(-iX) = SXDG to vertex a
        self.vops[a] = CliffordFrame::SXDG.compose(self.vops[a]);

        // Prepend sqrt(iZ) = SZ to each neighbor
        for &u in &nbrs {
            self.vops[u] = CliffordFrame::SZ.compose(self.vops[u]);
        }
    }

    // ========================================================================
    // Internal: CZ implementation
    // ========================================================================

    /// Check whether vertex `v` has any neighbor other than `other`.
    fn has_non_operand_neighbors(&self, v: usize, other: usize) -> bool {
        let nbrs = &self.neighbors[v];
        if nbrs.contains(other) {
            nbrs.len() >= 2
        } else {
            !nbrs.is_empty()
        }
    }

    /// Remove the VOP on vertex `v` by decomposing it into a sequence of
    /// local complementations on `v` and a chosen neighbor `vb`.
    ///
    /// Uses a precomputed decomposition table (BFS over the 24-element Clifford
    /// group using the two generators: LC on v appends SXDG, LC on vb appends SZ).
    fn remove_vop(&mut self, v: usize, avoid: usize) {
        use crate::clifford_frame::VOP_DECOMP;

        debug_assert!(
            !self.neighbors[v].is_empty(),
            "remove_vop called with isolated vertex"
        );

        // Pick a neighbor that isn't `avoid` (if possible)
        let mut vb = self.neighbors[v].iter().next().unwrap();
        if vb == avoid
            && let Some(alt) = self.neighbors[v].iter().find(|&u| u != avoid)
        {
            vb = alt;
        }
        // If avoid is the only neighbor, we'll use it anyway

        let (len, steps) = VOP_DECOMP[self.vops[v].index() as usize];

        // Apply steps in forward order: each step reduces the VOP toward identity
        for &step in &steps[..len as usize] {
            if step == 0 {
                // U: local complement on v
                self.local_complement(v);
            } else {
                // V: local complement on neighbor vb
                self.local_complement(vb);
            }
        }

        debug_assert!(
            self.vops[v].is_identity(),
            "remove_vop failed: VOP is {:?} (expected identity)",
            self.vops[v]
        );
    }

    /// Internal CZ implementation using the reference's 3-pass structure.
    ///
    /// Follows the Anders & Briegel `cphase` algorithm:
    /// 1. If v1 has non-operand neighbors, remove its VOP.
    /// 2. If v2 has non-operand neighbors, remove its VOP.
    /// 3. If v1 still has non-operand neighbors and non-diagonal VOP, remove again.
    /// 4. Apply CZ via lookup table.
    fn cz_internal(&mut self, v1: usize, v2: usize) {
        use crate::clifford_frame::CPHASE_TBL;

        if self.has_non_operand_neighbors(v1, v2) {
            self.remove_vop(v1, v2);
        }
        if self.has_non_operand_neighbors(v2, v1) {
            self.remove_vop(v2, v1);
        }
        if self.has_non_operand_neighbors(v1, v2) && !self.vops[v1].is_diagonal() {
            self.remove_vop(v1, v2);
        }

        // Use the CZ lookup table
        let was_edge = self.neighbors[v1].contains(v2);
        let op1 = self.vops[v1].index() as usize;
        let op2 = self.vops[v2].index() as usize;

        let we_idx = usize::from(was_edge);
        let [new_edge, new_op1, new_op2] = CPHASE_TBL[we_idx * 24 + op1][op2];

        // Set edge state
        let should_have_edge = new_edge == 1;
        if was_edge && !should_have_edge {
            // Remove edge
            self.neighbors[v1].toggle(v2);
            self.neighbors[v2].toggle(v1);
        } else if !was_edge && should_have_edge {
            // Add edge
            self.neighbors[v1].toggle(v2);
            self.neighbors[v2].toggle(v1);
        }

        self.vops[v1] = CliffordFrame::from_index(new_op1);
        self.vops[v2] = CliffordFrame::from_index(new_op2);
    }

    // ========================================================================
    // Internal: measurement
    // ========================================================================

    /// Measure qubit `a` in the Z basis with a given outcome for non-deterministic cases.
    ///
    /// Follows the reference's `measure` function: conjugate the Z basis through the VOP
    /// to determine which graph-state measurement to perform. If the conjugation produces
    /// a negative sign, flip the forced outcome before and the result after.
    fn measure_z_internal(&mut self, a: usize, forced_outcome: Option<bool>) -> MeasurementResult {
        // The effective Pauli being measured on the graph state
        let sigma = self.vops[a].z_image();
        let negative = !sigma.positive;

        // If the VOP conjugation gives a negative sign, flip the forced outcome
        let adjusted_forced = if negative {
            forced_outcome.map(|f| !f)
        } else {
            forced_outcome
        };

        let mut result = match sigma.axis {
            PauliAxis::X => self.measure_x_on_graph(a, adjusted_forced),
            PauliAxis::Y => self.measure_y_on_graph(a, adjusted_forced),
            PauliAxis::Z => self.measure_z_on_graph(a, adjusted_forced),
        };

        // If the sign was negative, flip the result
        if negative {
            result.outcome = !result.outcome;
        }

        result
    }

    /// Measure X on the graph state at vertex `v`.
    ///
    /// Follows the reference's `graph_X_measure` algorithm.
    /// If N(v) is empty: deterministic, outcome = 0 (always +1 eigenvalue).
    /// Otherwise: non-deterministic with 3-step edge toggling.
    fn measure_x_on_graph(&mut self, v: usize, forced_outcome: Option<bool>) -> MeasurementResult {
        if self.neighbors[v].is_empty() {
            // Deterministic: isolated graph state vertex is |+>, X eigenvalue +1
            return MeasurementResult {
                outcome: false,
                is_deterministic: true,
            };
        }

        // Non-deterministic
        let outcome = forced_outcome.unwrap_or_else(|| self.rng.coin_flip());

        // Pick a neighbor vb
        let vb = self.neighbors[v].iter().next().unwrap();

        // Save neighborhoods BEFORE modifications
        let vn: Vec<usize> = self.neighbors[v].iter().collect();
        let vbn: Vec<usize> = self.neighbors[vb].iter().collect();

        // Build sets for fast lookup
        let vn_set: BitSet = self.neighbors[v].clone();
        let vbn_set: BitSet = self.neighbors[vb].clone();

        // VOP updates
        if outcome {
            // Measured -1 (|->): SY on vb, Z on v, Z on N(vb) \ N(v) \ {v}
            self.vops[vb] = CliffordFrame::SY.compose(self.vops[vb]);
            self.vops[v] = CliffordFrame::Z.compose(self.vops[v]);
            for &u in &vbn {
                if u != v && !vn_set.contains(u) {
                    self.vops[u] = CliffordFrame::Z.compose(self.vops[u]);
                }
            }
        } else {
            // Measured +1 (|+>): SYDG on vb, Z on N(v) \ N(vb) \ {vb}
            self.vops[vb] = CliffordFrame::SYDG.compose(self.vops[vb]);
            for &u in &vn {
                if u != vb && !vbn_set.contains(u) {
                    self.vops[u] = CliffordFrame::Z.compose(self.vops[u]);
                }
            }
        }

        // Edge toggles (using saved neighborhoods)
        // STEP 1: Toggle edges between N(v) and N(vb), avoiding double-toggling
        {
            let mut processed = BitSet::new();
            for &i in &vn {
                for &j in &vbn {
                    if i != j {
                        let edge = if i < j { (i, j) } else { (j, i) };
                        let edge_key = edge.0 * self.num_qubits + edge.1;
                        if !processed.contains(edge_key) {
                            processed.insert(edge_key);
                            self.toggle_edge(i, j);
                        }
                    }
                }
            }
        }

        // STEP 2: Toggle complete subgraph on N(v) intersect N(vb)
        {
            let intersection: Vec<usize> = vn
                .iter()
                .filter(|&&u| vbn_set.contains(u))
                .copied()
                .collect();
            for i in 0..intersection.len() {
                for j in (i + 1)..intersection.len() {
                    self.toggle_edge(intersection[i], intersection[j]);
                }
            }
        }

        // STEP 3: Toggle edges from vb to N(v) \ {vb}
        for &u in &vn {
            if u != vb {
                self.toggle_edge(vb, u);
            }
        }

        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }

    /// Measure Y on the graph state at vertex `v`.
    ///
    /// Follows the reference's `graph_Y_measure` algorithm (direct, no reduction to X).
    /// Always non-deterministic.
    fn measure_y_on_graph(&mut self, v: usize, forced_outcome: Option<bool>) -> MeasurementResult {
        let outcome = forced_outcome.unwrap_or_else(|| self.rng.coin_flip());

        // Right-multiply each neighbor's VOP by SZDG (outcome=1) or SZ (outcome=0)
        let vnbg: Vec<usize> = self.neighbors[v].iter().collect();
        for &u in &vnbg {
            if outcome {
                self.vops[u] = CliffordFrame::SZDG.compose(self.vops[u]);
            } else {
                self.vops[u] = CliffordFrame::SZ.compose(self.vops[u]);
            }
        }

        // Toggle all edges in complete subgraph of {v} union N(v)
        let mut all_vertices = vnbg.clone();
        all_vertices.push(v);
        for i in 0..all_vertices.len() {
            for j in (i + 1)..all_vertices.len() {
                self.toggle_edge(all_vertices[i], all_vertices[j]);
            }
        }

        // Right-multiply v's VOP by SZ (outcome=0) or SZDG (outcome=1)
        if outcome {
            self.vops[v] = CliffordFrame::SZDG.compose(self.vops[v]);
        } else {
            self.vops[v] = CliffordFrame::SZ.compose(self.vops[v]);
        }

        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }

    /// Measure Z on the graph state at vertex `v`.
    ///
    /// Follows the reference's `graph_Z_measure` algorithm.
    /// Disconnects v from all neighbors (no edge complement among neighbors).
    /// If outcome=1, right-multiplies each neighbor's VOP by Z.
    /// Sets v's VOP by right-multiplying by H (outcome=0) or X*H=SY (outcome=1).
    fn measure_z_on_graph(&mut self, v: usize, forced_outcome: Option<bool>) -> MeasurementResult {
        let outcome = forced_outcome.unwrap_or_else(|| self.rng.coin_flip());

        let nbrs: Vec<usize> = self.neighbors[v].iter().collect();

        // Disconnect v from all neighbors (no edge complement)
        self.disconnect(v);

        // If outcome=1, right-multiply each neighbor's VOP by Z
        if outcome {
            for &u in &nbrs {
                self.vops[u] = CliffordFrame::Z.compose(self.vops[u]);
            }
        }

        // Set v's VOP: right-multiply by H (outcome=0) or X*H=SY (outcome=1)
        if outcome {
            // X * H = SY (index 10). Right-multiply: compose(SY, VOP) = VOP * SY
            self.vops[v] = CliffordFrame::SY.compose(self.vops[v]);
        } else {
            self.vops[v] = CliffordFrame::H.compose(self.vops[v]);
        }

        // Determine if deterministic: isolated vertices (no neighbors) have
        // deterministic Z measurement result. But after graph_Z_measure,
        // the result is always "non-deterministic" from the graph measurement
        // perspective. The determinism is handled by the caller (measure_z_internal).
        //
        // Actually: if the vertex had no neighbors to begin with, the graph state
        // X stabilizer means Z is non-deterministic.
        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }
}

// ============================================================================
// Trait implementations
// ============================================================================

impl<R: SeedableRng + Rng + Debug> QuantumSimulator for GraphStateSim<R> {
    fn reset(&mut self) -> &mut Self {
        // |0>^n = H^n |+>^n = H^n |G_empty>
        // So all VOPs are H, and the graph has no edges.
        for v in &mut self.vops {
            *v = CliffordFrame::H;
        }
        for n in &mut self.neighbors {
            n.clear();
        }
        self
    }
}

impl<R: SeedableRng + Rng + Debug> RngManageable for GraphStateSim<R> {
    type Rng = R;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.rng = rng;
    }

    #[inline]
    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    #[inline]
    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

impl<R: SeedableRng + Rng + Debug> CliffordGateable for GraphStateSim<R> {
    // -- Single-qubit gates: O(1) VOP composition --

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.vops[q.index()] = self.vops[q.index()].compose(CliffordFrame::X);
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.vops[q.index()] = self.vops[q.index()].compose(CliffordFrame::Y);
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.vops[q.index()] = self.vops[q.index()].compose(CliffordFrame::Z);
        }
        self
    }

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.vops[q.index()] = self.vops[q.index()].compose(CliffordFrame::SZ);
        }
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.vops[q.index()] = self.vops[q.index()].compose(CliffordFrame::SZDG);
        }
        self
    }

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.vops[q.index()] = self.vops[q.index()].compose(CliffordFrame::H);
        }
        self
    }

    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.vops[q.index()] = self.vops[q.index()].compose(CliffordFrame::SX);
        }
        self
    }

    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.vops[q.index()] = self.vops[q.index()].compose(CliffordFrame::SXDG);
        }
        self
    }

    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.vops[q.index()] = self.vops[q.index()].compose(CliffordFrame::SY);
        }
        self
    }

    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.vops[q.index()] = self.vops[q.index()].compose(CliffordFrame::SYDG);
        }
        self
    }

    // -- Two-qubit gates --

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(ctrl, targ) in pairs {
            let ctrl = ctrl.index();
            let targ = targ.index();
            // CX = (I x H) CZ (I x H)
            self.vops[targ] = self.vops[targ].compose(CliffordFrame::H);
            self.cz_internal(ctrl, targ);
            self.vops[targ] = self.vops[targ].compose(CliffordFrame::H);
        }
        self
    }

    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            self.cz_internal(q0.index(), q1.index());
        }
        self
    }

    // -- Measurement --

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits
            .iter()
            .map(|&q| self.measure_z_internal(q.index(), None))
            .collect()
    }
}

// ============================================================================
// ForcedMeasurement & StabilizerSimulator
// ============================================================================

impl<R: SeedableRng + Rng + Debug> ForcedMeasurement for GraphStateSim<R> {
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        self.measure_z_internal(qubit, Some(forced_outcome))
    }
}

impl<R: SeedableRng + Rng + Debug> crate::StabilizerTableauSimulator for GraphStateSim<R> {
    fn stab_tableau(&self) -> String {
        let gs = self.to_graph_state();
        let n = gs.num_qubits();
        let gens = gs.stabilizer_generators();
        let mut result = String::with_capacity(n * (n + 3));
        for g in &gens {
            pauli_string_to_tableau_line(g, n, &mut result);
        }
        result
    }

    fn destab_tableau(&self) -> String {
        let n = self.num_qubits;
        let mut result = String::with_capacity(n * (n + 3));
        for v in 0..n {
            let z_img = self.vops[v].z_image();
            let pauli = match z_img.axis {
                PauliAxis::X => pecos_core::Pauli::X,
                PauliAxis::Y => pecos_core::Pauli::Y,
                PauliAxis::Z => pecos_core::Pauli::Z,
            };
            let phase = if z_img.positive {
                pecos_core::QuarterPhase::PlusOne
            } else {
                pecos_core::QuarterPhase::MinusOne
            };
            let mut paulis = vec![pecos_core::Pauli::I; n];
            paulis[v] = pauli;
            let ps = pecos_core::PauliString::from_paulis_with_phase(phase, &paulis);
            pauli_string_to_tableau_line(&ps, n, &mut result);
        }
        result
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

/// Format a `PauliString` as a tableau line matching the `DenseStab` format.
///
/// Produces e.g. `"+ZI\n"` or `"-iXY\n"`.
fn pauli_string_to_tableau_line(ps: &pecos_core::PauliString, n: usize, out: &mut String) {
    use std::fmt::Write;
    let phase_str = match ps.phase() {
        pecos_core::QuarterPhase::PlusOne => "+",
        pecos_core::QuarterPhase::MinusOne => "-",
        pecos_core::QuarterPhase::PlusI => "+i",
        pecos_core::QuarterPhase::MinusI => "-i",
    };
    writeln!(out, "{}{}", phase_str, ps.pauli_str(Some(n))).unwrap();
}

impl StabilizerSimulator for GraphStateSim<PecosRng> {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stabilizer_test_suite;
    use pecos_core::qid;

    stabilizer_test_suite!(GraphStateSim);

    #[test]
    fn test_initial_state_is_all_zero() {
        let mut sim = GraphStateSim::with_seed(3, 42);
        for i in 0..3 {
            let result = sim.mz(&[QubitId::new(i)]);
            assert!(
                result[0].is_deterministic,
                "qubit {i} should be deterministic"
            );
            assert!(!result[0].outcome, "qubit {i} should be |0>");
        }
    }

    #[test]
    fn test_single_qubit_x_flips() {
        let mut sim = GraphStateSim::with_seed(1, 42);
        sim.x(&qid(0));
        let result = sim.mz(&qid(0));
        assert!(result[0].is_deterministic);
        assert!(result[0].outcome, "X|0> = |1>");
    }

    #[test]
    fn test_hadamard_creates_superposition() {
        let mut sim = GraphStateSim::with_seed(1, 42);
        sim.h(&qid(0));
        let result = sim.mz(&qid(0));
        assert!(
            !result[0].is_deterministic,
            "H|0> = |+> should be non-deterministic for mz"
        );
    }

    #[test]
    fn test_bell_state_correlations() {
        // Create Bell state and verify correlations over many seeds
        for seed in 0..20 {
            let mut sim = GraphStateSim::with_seed(2, seed);
            sim.h(&qid(0));
            sim.cx(&[(QubitId::new(0), QubitId::new(1))]);

            let r0 = sim.mz(&qid(0));
            let r1 = sim.mz(&qid(1));
            assert!(!r0[0].is_deterministic);
            assert!(
                r1[0].is_deterministic,
                "second qubit should be deterministic after first measured"
            );
            assert_eq!(
                r0[0].outcome, r1[0].outcome,
                "Bell state qubits should be correlated"
            );
        }
    }

    #[test]
    fn test_cz_creates_cluster_state() {
        let mut sim = GraphStateSim::with_seed(2, 42);
        sim.h(&qid(0));
        sim.h(&[QubitId::new(1)]);
        sim.cz(&[(QubitId::new(0), QubitId::new(1))]);

        // CZ|++> should give a 2-qubit cluster state
        // Measuring Z on qubit 0 should be non-deterministic
        let r = sim.mz(&qid(0));
        assert!(!r[0].is_deterministic);
    }

    #[test]
    fn test_ghz_state() {
        for seed in 0..20 {
            let mut sim = GraphStateSim::with_seed(3, seed);
            sim.h(&qid(0));
            sim.cx(&[(QubitId::new(0), QubitId::new(1))]);
            sim.cx(&[(QubitId::new(1), QubitId::new(2))]);

            let r0 = sim.mz(&qid(0));
            let r1 = sim.mz(&[QubitId::new(1)]);
            let r2 = sim.mz(&[QubitId::new(2)]);

            assert!(!r0[0].is_deterministic);
            assert_eq!(r0[0].outcome, r1[0].outcome, "GHZ: q0 == q1");
            assert_eq!(r1[0].outcome, r2[0].outcome, "GHZ: q1 == q2");
        }
    }

    #[test]
    fn test_measurement_idempotent() {
        let mut sim = GraphStateSim::with_seed(1, 42);
        sim.h(&qid(0));
        let r1 = sim.mz(&qid(0));
        let r2 = sim.mz(&qid(0));
        assert!(
            r2[0].is_deterministic,
            "second measurement should be deterministic"
        );
        assert_eq!(
            r1[0].outcome, r2[0].outcome,
            "repeated measurement should give same result"
        );
    }

    #[test]
    fn test_sz_gate() {
        let mut sim = GraphStateSim::with_seed(1, 42);
        // SZ SZ = Z, and Z|0> = |0>
        sim.sz(&qid(0));
        sim.sz(&qid(0));
        let result = sim.mz(&qid(0));
        assert!(result[0].is_deterministic);
        assert!(!result[0].outcome, "Z|0> = |0>");
    }

    #[test]
    fn test_cross_validation_random_circuits() {
        use crate::SparseStab;
        use crate::stabilizer_test_utils::compare_simulators_on_random_circuits_direct;

        let mut gs = GraphStateSim::with_seed(6, 0);
        let mut ss = SparseStab::with_seed(6, 0);
        compare_simulators_on_random_circuits_direct(&mut gs, &mut ss, 6, 30, 50, 98765);
    }
}
