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

/// Compile-only STN analyzer for replaying a circuit before choosing a simulator.
///
/// Replay the same gates that an execution simulator would receive, then call
/// [`Self::recommend`] or [`Self::advise`]. The analysis tracks the Clifford
/// tableau and OFD-relevant GF(2) flip patterns without allocating an MPS.
/// Recommendations are heuristic dispatch guidance, not resource or runtime
/// guarantees.
///
/// ```
/// use pecos_core::{Angle64, QubitId};
/// use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
/// use pecos_stab_tn::stab_mps::compile::{InjectionMode, StabMpsCompile};
///
/// let mut analysis = StabMpsCompile::new(20);
/// analysis.h(&[QubitId(0)]);
/// analysis.cx(&[(QubitId(0), QubitId(1))]);
/// analysis.rz(Angle64::QUARTER_TURN / 2_u64, &[QubitId(0)]);
///
/// let recommendation = analysis.recommend();
/// let advice = analysis.advise(Some(analysis.nonclifford_rz_total() as usize));
/// assert_eq!(advice.injection, InjectionMode::Deferred);
/// println!("{:?}: {}", recommendation.kind, recommendation.reason);
/// ```
///
/// [`Self::recommend`] applies these rules in order: pure Clifford selects
/// `CHForm`; otherwise `n <= 14` selects a dense state vector; otherwise OFD
/// nullity `<= 6` selects `StabMps`; otherwise non-Clifford count `<= 40`
/// selects `StabVec`; all remaining circuits select `StabMps` with adaptive
/// bond growth suggested. [`Self::bond_dim_bound`] returns `2^nullity` when
/// representable and saturates at [`usize::MAX`] if that power overflows.
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
    /// Number of non-Clifford RZ gates processed.
    nonclifford_rz_total: u64,
    /// Number whose deferred-injection correction RZ(2*theta) is Clifford.
    injectable_clifford_correction: u64,
}

impl StabMpsCompile {
    /// Create an empty compile-only analysis for `num_qubits` qubits.
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
            nonclifford_rz_total: 0,
            injectable_clifford_correction: 0,
        }
    }

    #[must_use]
    /// Return the number of qubits being analyzed.
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

    /// Total non-Clifford RZ gates processed.
    #[must_use]
    pub fn nonclifford_rz_total(&self) -> u64 {
        self.nonclifford_rz_total
    }

    /// Number of non-Clifford RZ gates whose RZ(2*theta) injection
    /// correction is Clifford.
    #[must_use]
    pub fn injectable_clifford_correction(&self) -> u64 {
        self.injectable_clifford_correction
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

    /// Theoretical bond-dimension upper bound, `2^nullity`.
    ///
    /// Returns one at zero nullity and saturates at [`usize::MAX`] when the
    /// power of two cannot be represented by `usize` on the target platform.
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

    /// Heuristically recommend a PECOS simulator for the accumulated circuit.
    ///
    /// Rules are evaluated in order: pure Clifford selects `CHForm`; otherwise
    /// `n <= 14` selects `StateVector`; otherwise nullity `<= 6` selects
    /// `StabMps`; otherwise total non-Clifford count `<= 40` selects `StabVec`;
    /// all remaining cases select `StabMps` and suggest adaptive bond growth.
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

    /// Advise a simulator and magic-state injection mode for the accumulated circuit.
    ///
    /// `ancilla_budget` is the number of fresh ancillas available for deferred
    /// injection. `None` leaves feasibility unspecified and emits a warning;
    /// a sufficient budget selects deferred injection for injectable gates;
    /// an insufficient nonzero budget selects immediate injection; zero selects
    /// direct application. Required deferred capacity is one ancilla for every
    /// non-Clifford RZ, including arbitrary-angle rotations whose correction is
    /// itself non-Clifford.
    #[must_use]
    pub fn advise(&self, ancilla_budget: Option<usize>) -> ExecutionAdvice {
        let base = self.recommend();
        let injectable_count = self.injectable_clifford_correction();
        // Mast consumes one fresh ancilla per non-Clifford RZ regardless of
        // angle (mast.rs inject_magic_state); only the correction's
        // Cliffordness depends on injectability. Budget feasibility must
        // therefore count every non-Clifford RZ, not only T-like ones.
        let deferred_ancillas_required = self.nonclifford_rz_total();
        let deferred_feasible = ancilla_budget.map(|budget| {
            usize::try_from(deferred_ancillas_required).is_ok_and(|required| budget >= required)
        });
        let mut warnings = Vec::new();

        let injection = if injectable_count == 0 {
            InjectionMode::Direct
        } else {
            match ancilla_budget {
                Some(0) => InjectionMode::Direct,
                Some(_) if deferred_feasible == Some(true) => InjectionMode::Deferred,
                None => {
                    warnings.push(format!(
                        "ancilla budget was unspecified; deferred injection requires \
                     {deferred_ancillas_required} fresh ancilla(s)"
                    ));
                    InjectionMode::Deferred
                }
                Some(budget) => {
                    warnings.push(format!(
                        "deferred injection needs one fresh ancilla per non-Clifford RZ \
                         ({deferred_ancillas_required} required); the given budget of {budget} is \
                         insufficient"
                    ));
                    InjectionMode::Immediate
                }
            }
        };

        let noninjectable_count = self.nonclifford_rz_total().saturating_sub(injectable_count);
        if noninjectable_count > 0 {
            warnings.push(format!(
                "{noninjectable_count} non-injectable non-Clifford RZ gate(s) have arbitrary \
                 angles; deferred projection pays a non-Clifford correction per hit for those \
                 gates"
            ));
        }

        let simulator = if base.kind == SimulatorKind::StabMps
            && injectable_count > 0
            && deferred_feasible != Some(false)
        {
            SimulatorKind::Mast
        } else {
            base.kind
        };

        let mode_reason = match injection {
            InjectionMode::Direct => "direct non-Clifford application advised",
            InjectionMode::Immediate => {
                "immediate magic-state injection advised with one reusable ancilla"
            }
            InjectionMode::Deferred => {
                "deferred magic-state injection advised to keep the coefficient MPS near bond 1"
            }
        };

        ExecutionAdvice {
            simulator,
            injection,
            injectable_count,
            deferred_ancillas_required,
            deferred_feasible,
            warnings,
            reason: format!("{}; {mode_reason}", base.reason),
        }
    }

    /// Process one non-Clifford Z-rotation on qubit q. Mirrors the decision
    /// logic of `non_clifford::apply_rz_stab_mps` but does not modify any MPS.
    fn process_rz(&mut self, theta: Angle64, q: usize) {
        self.nonclifford_rz_total += 1;
        if is_clifford_rz(theta + theta) {
            self.injectable_clifford_correction += 1;
        }

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

fn is_clifford_rz(theta: Angle64) -> bool {
    theta == Angle64::ZERO
        || theta == Angle64::HALF_TURN
        || theta == Angle64::QUARTER_TURN
        || theta == Angle64::THREE_QUARTERS_TURN
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
    /// Magic-state injection Augmented Stabilizer Tensor Network
    /// (`pecos_stab_tn::stab_mps::mast::Mast`). Deferred magic-state
    /// injection over the expanded data+ancilla register. Best for
    /// Clifford+T-like circuits where deferral keeps the coefficient MPS
    /// near bond 1.
    Mast,
}

/// Heuristic simulator recommendation returned by [`StabMpsCompile::recommend`].
#[derive(Clone, Debug)]
pub struct SimulatorRecommendation {
    /// Simulator selected by the ordered recommendation thresholds.
    pub kind: SimulatorKind,
    /// Human-readable rule and observed metric that led to the selection.
    pub reason: String,
}

/// Strategy for applying injectable non-Clifford gates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InjectionMode {
    /// Apply non-Clifford gates directly without magic-state injection.
    Direct,
    /// Inject immediately using one reusable ancilla.
    Immediate,
    /// Defer magic-state projections using one fresh ancilla per gate.
    Deferred,
}

/// Typed simulator and injection recommendation from compile-only analysis.
#[derive(Clone, Debug)]
pub struct ExecutionAdvice {
    /// Recommended simulator.
    pub simulator: SimulatorKind,
    /// Recommended non-Clifford injection strategy.
    pub injection: InjectionMode,
    /// Number of gates with Clifford deferred-injection corrections.
    pub injectable_count: u64,
    /// Fresh ancillas required for deferred injection: one per non-Clifford
    /// RZ (injectable or not — Mast injects for every non-Clifford gate).
    pub deferred_ancillas_required: u64,
    /// Whether the supplied budget supports deferral, or `None` if unspecified.
    pub deferred_feasible: Option<bool>,
    /// Non-fatal qualifications of the advice.
    pub warnings: Vec<String>,
    /// Human-readable explanation of the recommendation.
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
        self.nonclifford_rz_total = 0;
        self.injectable_clifford_correction = 0;
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
            self.process_rz(theta, q.index());
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

    #[test]
    fn test_compile_counts_injectable_rz_corrections() {
        let mut comp = StabMpsCompile::new(8);
        let t = Angle64::QUARTER_TURN / 2u64;
        let tdg = -t;

        comp.h(&[QubitId(0)]);
        comp.rz(t, &[QubitId(0), QubitId(1), QubitId(2)]);
        comp.rz(tdg, &[QubitId(3)]);
        comp.rz(Angle64::from_radians(0.3), &[QubitId(4), QubitId(5)]);
        comp.sz(&[QubitId(6)]);

        assert_eq!(comp.injectable_clifford_correction(), 4);
        assert_eq!(comp.nonclifford_rz_total(), 6);
        assert_eq!(comp.total_nonclifford(), 6);

        comp.reset();
        assert_eq!(comp.injectable_clifford_correction(), 0);
        assert_eq!(comp.nonclifford_rz_total(), 0);
    }

    fn compile_t_gates(count: usize) -> StabMpsCompile {
        let mut comp = StabMpsCompile::new(20);
        let t = Angle64::QUARTER_TURN / 2u64;
        for _ in 0..count {
            comp.rz(t, &[QubitId(0)]);
        }
        comp
    }

    #[test]
    fn test_advise_no_injectables_uses_direct_application() {
        let mut comp = StabMpsCompile::new(20);
        comp.rz(Angle64::from_radians(0.3), &[QubitId(0)]);

        let advice = comp.advise(Some(4));
        assert_eq!(advice.simulator, SimulatorKind::StabMps);
        assert_eq!(advice.injection, InjectionMode::Direct);
        assert_eq!(advice.injectable_count, 0);
        // Every non-Clifford RZ consumes a fresh ancilla under deferral,
        // injectable or not.
        assert_eq!(advice.deferred_ancillas_required, 1);
        assert_eq!(advice.deferred_feasible, Some(true));
    }

    #[test]
    fn test_advise_mixed_angles_count_all_nonclifford_rz_for_deferral() {
        // One T (injectable) plus one arbitrary-angle RZ: deferral consumes
        // two ancillas. A budget covering only the injectable gate must not
        // be reported feasible for deferred execution.
        let mut comp = compile_t_gates(1);
        comp.rz(Angle64::from_radians(0.3), &[QubitId(0)]);

        let advice = comp.advise(Some(1));
        assert_eq!(advice.injectable_count, 1);
        assert_eq!(advice.deferred_ancillas_required, 2);
        assert_eq!(advice.deferred_feasible, Some(false));
        assert_ne!(advice.injection, InjectionMode::Deferred);

        let advice = comp.advise(Some(2));
        assert_eq!(advice.deferred_feasible, Some(true));
        assert_eq!(advice.injection, InjectionMode::Deferred);
    }

    #[test]
    fn test_advise_zero_budget_uses_direct_application() {
        let advice = compile_t_gates(1).advise(Some(0));

        assert_eq!(advice.simulator, SimulatorKind::StabMps);
        assert_eq!(advice.injection, InjectionMode::Direct);
        assert_eq!(advice.deferred_feasible, Some(false));
    }

    #[test]
    fn test_advise_unspecified_budget_selects_mast_with_warning() {
        let advice = compile_t_gates(2).advise(None);

        assert_eq!(advice.simulator, SimulatorKind::Mast);
        assert_eq!(advice.injection, InjectionMode::Deferred);
        assert_eq!(advice.injectable_count, 2);
        assert_eq!(advice.deferred_ancillas_required, 2);
        assert_eq!(advice.deferred_feasible, None);
        assert!(
            advice
                .warnings
                .iter()
                .any(|warning| warning.contains("budget was unspecified"))
        );
        assert!(advice.reason.contains("low OFD nullity"));
        assert!(advice.reason.contains("STN bond dim bound"));
    }

    #[test]
    fn test_advise_sufficient_budget_selects_mast() {
        let advice = compile_t_gates(2).advise(Some(2));

        assert_eq!(advice.simulator, SimulatorKind::Mast);
        assert_eq!(advice.injection, InjectionMode::Deferred);
        assert_eq!(advice.deferred_feasible, Some(true));
        assert!(advice.warnings.is_empty());
    }

    #[test]
    fn test_advise_insufficient_budget_uses_immediate_injection() {
        let advice = compile_t_gates(2).advise(Some(1));

        assert_eq!(advice.simulator, SimulatorKind::StabMps);
        assert_eq!(advice.injection, InjectionMode::Immediate);
        assert_eq!(advice.deferred_feasible, Some(false));
        assert!(advice.warnings.iter().any(|warning| {
            warning.contains("one fresh ancilla per non-Clifford RZ")
                && warning.contains("budget of 1 is insufficient")
        }));
    }

    #[test]
    fn test_advise_warns_about_arbitrary_angle_corrections() {
        let mut comp = compile_t_gates(1);
        comp.rz(Angle64::from_radians(0.3), &[QubitId(1)]);

        let advice = comp.advise(Some(1));
        assert!(advice.warnings.iter().any(|warning| {
            warning.contains("non-injectable non-Clifford RZ")
                && warning.contains("non-Clifford correction per hit")
        }));
    }

    #[test]
    fn test_advise_preserves_non_stn_base_simulator() {
        let mut comp = StabMpsCompile::new(8);
        comp.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);

        let advice = comp.advise(Some(1));
        assert_eq!(advice.simulator, SimulatorKind::StateVector);
        assert_eq!(advice.injection, InjectionMode::Deferred);
    }
}
