// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Compile-only pre-analysis for STN tractability.
//!
//! Runs through a circuit's Clifford tableau and non-Clifford gate decomposition
//! WITHOUT building an MPS. Reports the GF(2) nullity of the accumulated flip
//! patterns, which per Liu-Clark 2412.17209 bounds the CAMPS bond dimension:
//!   `bond_dim` ≤ 2^nullity.
//!
//! Useful for deciding whether a circuit is tractable for full simulation
//! before committing resources. Complexity is O(t·n²) for t non-Cliffords
//! and n qubits (Clifford tableau ops dominate).

use super::ofd::{Gf2FlipMatrix, RowMetadata};
use super::pauli_decomp::{ZDecomposition, decompose_z};
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, MeasurementResult, QuantumSimulator, SparseStabY,
};

/// Compile-only STN analyzer: runs Clifford tableau and tracks OFD-relevant
/// GF(2) flip patterns, without any MPS representation.
pub struct StabMpsCompile {
    num_qubits: usize,
    tableau: SparseStabY,
    gf2_matrix: Gf2FlipMatrix,
    /// Per-site "free qubit" flag: true if this qubit has never been the
    /// disent `rot_site`. Mirrors our `disent_flags` for OFD applicability.
    free_qubit: Vec<bool>,
    /// Number of non-Clifford gates that OFD would absorb (consume a free qubit).
    absorbed: u64,
    /// Number of non-Clifford gates that would grow bond dim.
    grown: u64,
    /// Number of non-Cliffords that hit the Stabilizer branch (no MPS site op).
    stabilizer: u64,
}

impl StabMpsCompile {
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            tableau: SparseStabY::new(num_qubits).with_destab_sign_tracking(),
            gf2_matrix: Gf2FlipMatrix::new(num_qubits),
            free_qubit: vec![true; num_qubits],
            absorbed: 0,
            grown: 0,
            stabilizer: 0,
        }
    }

    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of non-Clifford gates that consumed a free qubit (disentangled).
    #[must_use]
    pub fn absorbed(&self) -> u64 {
        self.absorbed
    }

    /// Number of non-Clifford gates that would grow bond dim.
    #[must_use]
    pub fn grown(&self) -> u64 {
        self.grown
    }

    /// Number of non-Cliffords that hit the Stabilizer branch.
    #[must_use]
    pub fn stabilizer(&self) -> u64 {
        self.stabilizer
    }

    /// Total non-Clifford gates processed.
    #[must_use]
    pub fn total_nonclifford(&self) -> u64 {
        self.absorbed + self.grown + self.stabilizer
    }

    /// GF(2) nullity = number of flip patterns NOT in the rank.
    /// Bond dim bound from OFD is 2^nullity.
    #[must_use]
    pub fn nullity(&self) -> usize {
        let t = self.gf2_matrix.num_gates();
        t.saturating_sub(self.gf2_matrix.gf2_rank())
    }

    /// Rank of accumulated GF(2) matrix.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.gf2_matrix.gf2_rank()
    }

    /// Theoretical bond dim upper bound: 2^nullity.
    #[must_use]
    pub fn bond_dim_bound(&self) -> usize {
        let n = self.nullity();
        if n == 0 {
            1
        } else {
            1usize
                .checked_shl(u32::try_from(n).unwrap_or(u32::MAX))
                .unwrap_or(usize::MAX)
        }
    }

    /// Access the accumulated GF(2) matrix for inspection.
    #[must_use]
    pub fn gf2_matrix(&self) -> &Gf2FlipMatrix {
        &self.gf2_matrix
    }

    /// Recommend which PECOS simulator best fits the accumulated circuit
    /// characteristics. Based on a heuristic cost model — see the
    /// `SimulatorRecommendation` docstring for exact decision rules.
    ///
    /// Use case: after running a circuit through `StabMpsCompile` (which does
    /// O(t·n²) pre-analysis without any MPS overhead), dispatch to the
    /// best simulator for actual simulation.
    #[must_use]
    pub fn recommend(&self) -> SimulatorRecommendation {
        let n = self.num_qubits();
        let t = self.total_nonclifford();
        let nullity = self.nullity();

        // Pure Clifford: CHForm is exact and fastest.
        if t == 0 {
            return SimulatorRecommendation {
                kind: SimulatorKind::CHForm,
                reason: "pure Clifford circuit — CHForm is exact and O(n²) memory".to_string(),
            };
        }
        // Small n: dense state vector is straightforward and fastest.
        if n <= 14 {
            return SimulatorRecommendation {
                kind: SimulatorKind::StateVector,
                reason: format!("small system (n={n} ≤ 14) — dense state vector fits in memory"),
            };
        }
        // Low-rank: STN bond dim bound is 2^nullity; stays cheap at small nullity.
        if nullity <= 6 {
            return SimulatorRecommendation {
                kind: SimulatorKind::StabMps,
                reason: format!(
                    "low OFD nullity ({nullity}) — STN bond dim bound 2^{nullity} = {}",
                    1usize << nullity
                ),
            };
        }
        // Moderate T-count: StabVec stabilizer-sum with pruning.
        if t <= 40 {
            return SimulatorRecommendation {
                kind: SimulatorKind::StabVec,
                reason: format!("moderate T-count (t={t} ≤ 40) — StabVec with MC pruning"),
            };
        }
        // Fallback: STN with adaptive bond-dim cap.
        SimulatorRecommendation {
            kind: SimulatorKind::StabMps,
            reason: format!(
                "large nullity (nullity={nullity}) and high T-count (t={t}) — \
                 STN with auto_grow_bond_dim recommended"
            ),
        }
    }

    /// Process one non-Clifford Z-rotation on qubit q. Mirrors the decision
    /// logic of `non_clifford::apply_rz_stab_mps` but does not modify any MPS.
    fn process_rz(&mut self, q: usize) {
        let decomp = decompose_z(self.tableau.stabs(), self.tableau.destabs(), q);
        match decomp {
            ZDecomposition::Stabilizer { .. } => {
                self.stabilizer += 1;
            }
            ZDecomposition::DestabilizerFlip {
                ref flip_sites,
                ref sign_sites,
                ..
            } => {
                // Build list of affected sites (union of flip + sign).
                let mut sites: std::collections::BTreeSet<usize> =
                    std::collections::BTreeSet::new();
                for s in flip_sites {
                    sites.insert(*s);
                }
                for s in sign_sites {
                    sites.insert(*s);
                }
                let affected: Vec<usize> = sites.into_iter().collect();

                if affected.len() == 1 {
                    // Single-site path: always absorbable.
                    let site = affected[0];
                    self.absorbed += 1;
                    let flip_vec: Vec<usize> = flip_sites.clone();
                    self.gf2_matrix
                        .add_row_with_meta(&flip_vec, RowMetadata { rot_site: site });
                    self.free_qubit[site] = false;
                } else {
                    // Multi-site: OFD condition is "some site i has free_qubit[i]
                    // AND site i has X/Y pauli (i.e. i ∈ flip_sites)".
                    let mut rot = None;
                    for &s in &affected {
                        if self.free_qubit[s] && flip_sites.contains(&s) {
                            rot = Some(s);
                            break;
                        }
                    }
                    if let Some(site) = rot {
                        self.absorbed += 1;
                        let flip_vec: Vec<usize> = flip_sites.clone();
                        self.gf2_matrix
                            .add_row_with_meta(&flip_vec, RowMetadata { rot_site: site });
                        self.free_qubit[site] = false;
                    } else {
                        self.grown += 1;
                    }
                }
            }
        }
    }
}

/// Classification of PECOS simulators for dispatch purposes.
/// See `StabMpsCompile::recommend`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimulatorKind {
    /// Dense state vector (e.g., `pecos_simulators::StateVec`). Exact;
    /// O(2^n) memory. Best for small n.
    StateVector,
    /// CH-form stabilizer simulator
    /// (`pecos_simulators::CHForm`). Exact for pure Clifford; O(n²) memory.
    CHForm,
    /// Clifford+Rz stabilizer-sum simulator
    /// (`pecos_simulators::StabVec`). Stabilizer-rank method with
    /// MC pruning. Best for moderate T-count.
    StabVec,
    /// Stabilizer Tensor Network
    /// (`pecos_stab_tn::stab_mps::StabMps`). Hybrid tableau+MPS. Best for
    /// low-rank (low OFD nullity) circuits and T-heavy circuits with
    /// adaptive bond-dim.
    StabMps,
}

/// Simulator recommendation with a human-readable reason string.
/// Returned by `StabMpsCompile::recommend`.
#[derive(Clone, Debug)]
pub struct SimulatorRecommendation {
    pub kind: SimulatorKind,
    pub reason: String,
}

impl QuantumSimulator for StabMpsCompile {
    fn reset(&mut self) -> &mut Self {
        self.tableau = SparseStabY::new(self.num_qubits).with_destab_sign_tracking();
        self.gf2_matrix.reset();
        self.free_qubit = vec![true; self.num_qubits];
        self.absorbed = 0;
        self.grown = 0;
        self.stabilizer = 0;
        self
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

impl CliffordGateable for StabMpsCompile {
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.tableau.sz(qubits);
        self
    }
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.tableau.h(qubits);
        self
    }
    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.tableau.cx(pairs);
        self
    }
    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.tableau.cz(pairs);
        self
    }
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        // Compile mode: delegate to tableau (no MPS needed for measurement).
        self.tableau.mz(qubits)
    }
}

impl ArbitraryRotationGateable for StabMpsCompile {
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        self.h(qubits);
        self.rz(theta, qubits);
        self.h(qubits);
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            // Handle Clifford angles as Cliffords.
            if theta == Angle64::ZERO {
                continue;
            }
            if theta == Angle64::HALF_TURN {
                self.tableau.z(&[q]);
                continue;
            }
            if theta == Angle64::QUARTER_TURN {
                self.tableau.sz(&[q]);
                continue;
            }
            if theta == Angle64::THREE_QUARTERS_TURN {
                self.tableau.szdg(&[q]);
                continue;
            }
            // Non-Clifford: process decomposition.
            self.process_rz(q.index());
        }
        self
    }

    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            self.cx(&[(q0, q1)]);
            self.rz(theta, &[q1]);
            self.cx(&[(q0, q1)]);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_sizes() {
        let s = StabMpsCompile::new(5);
        assert_eq!(s.num_qubits(), 5);
        assert_eq!(s.nullity(), 0);
        assert_eq!(s.bond_dim_bound(), 1);
    }

    #[test]
    fn test_compile_all_independent_t_gates() {
        let mut s = StabMpsCompile::new(5);
        s.h(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3), QubitId(4)]);
        for i in 0..5 {
            s.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(i)]);
        }
        assert_eq!(s.absorbed(), 5);
        assert_eq!(s.grown(), 0);
        assert_eq!(s.nullity(), 0);
        assert_eq!(s.bond_dim_bound(), 1);
    }

    #[test]
    fn test_compile_vs_stn_nullity_matches() {
        // Verify that StabMpsCompile and full StabMps agree on nullity for same circuit.
        use crate::stab_mps::StabMps;
        let q = |i: usize| QubitId(i);
        let mut comp = StabMpsCompile::new(4);
        comp.h(&[q(0), q(1), q(2), q(3)]);
        comp.rz(Angle64::QUARTER_TURN / 2u64, &[q(0)]);
        comp.cx(&[(q(0), q(1))]);
        comp.rz(Angle64::QUARTER_TURN / 2u64, &[q(1)]);
        comp.rz(Angle64::QUARTER_TURN / 2u64, &[q(2)]);

        let mut stn = StabMps::with_seed(4, 1);
        stn.h(&[q(0), q(1), q(2), q(3)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(0)]);
        stn.cx(&[(q(0), q(1))]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(1)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(2)]);

        assert_eq!(
            comp.nullity(),
            stn.ofd_nullity(),
            "StabMpsCompile and StabMps should report same OFD nullity"
        );
    }

    #[test]
    fn test_recommend_pure_clifford_prefers_chform() {
        let mut comp = StabMpsCompile::new(4);
        comp.h(&[QubitId(0), QubitId(1)]);
        comp.cx(&[(QubitId(0), QubitId(1))]);
        // t = 0.
        let r = comp.recommend();
        assert_eq!(r.kind, SimulatorKind::CHForm);
    }

    #[test]
    fn test_recommend_small_n_prefers_state_vector() {
        let mut comp = StabMpsCompile::new(8);
        comp.h(&[QubitId(0)]);
        comp.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]); // one T
        let r = comp.recommend();
        assert_eq!(r.kind, SimulatorKind::StateVector);
    }

    #[test]
    fn test_recommend_low_nullity_prefers_stn() {
        let n = 20;
        let mut comp = StabMpsCompile::new(n);
        // Simple Clifford + independent T gates (nullity = 0 because
        // same flip pattern on unique qubits each rank-1).
        // H on qubit 0, T on qubit 0 gives one flip pattern of weight 1.
        // Multiple independent Ts → independent flip patterns → all rank,
        // zero nullity.
        comp.h(&[QubitId(0)]);
        comp.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        let r = comp.recommend();
        assert_eq!(
            r.kind,
            SimulatorKind::StabMps,
            "nullity={} should recommend STN for n={n} (reason: {})",
            comp.nullity(),
            r.reason
        );
    }
}
