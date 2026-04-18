// Copyright 2024 The PECOS Developers
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

//! Y-convention sparse stabilizer simulator.
//!
//! In this convention, both X and Z bits set represents Y = iXZ directly,
//! rather than W = XZ as in `SparseStab`. This means gate operations only
//! need +/-1 phases (no `signs_i` during gates). The `signs_i` field is
//! still present in `GensGeneric` and used during measurement where
//! anticommuting Hermitian products can produce anti-Hermitian results.
//!
//! # Sign rules (Y-convention vs W-convention)
//!
//! - X, Y, Z, H: identical to W-convention
//! - SZ: `signs_minus ^= col_x[q] AND col_z[q]` (simpler; W needs `signs_i`)
//! - CX: `signs_minus ^= col_x[q1] AND col_z[q2] AND NOT(col_z[q1] XOR col_x[q2])`
//!   (W-convention has no sign update for CX)
//! - Measurement: needs additional `n_Y` accounting via `signs_i`

use crate::{CliffordGateable, GensGeneric, MeasurementResult, QuantumSimulator};
use core::fmt::Debug;
use core::mem;
use pecos_core::{BitSet, IndexSet, QubitId, RngManageable, SortedVecSet, VecSet};
use pecos_random::rng_ext::RngProbabilityExt;
use pecos_random::{PecosRng, Rng, SeedableRng};

/// Y-convention sparse stabilizer simulator.
///
/// Same structure as `SparseStabGeneric` but with Y-convention sign rules:
/// both X and Z bits set = Y = iXZ (not W = XZ).
#[derive(Clone, Debug)]
pub struct SparseStabYGeneric<S: IndexSet = BitSet, R: SeedableRng + Rng + Debug = PecosRng> {
    pub(crate) num_qubits: usize,
    pub(crate) stabs: GensGeneric<S>,
    pub(crate) destabs: GensGeneric<S>,
    pub(crate) rng: R,
    /// When true, maintain destabilizer signs through Clifford gates.
    /// Off by default (not needed for standard stabilizer simulation).
    /// Required for STN-style decomposition that uses destabilizer phases.
    track_destab_signs: bool,
}

/// Default Y-convention sparse stabilizer simulator using `BitSet`.
pub type SparseStabY<R = PecosRng> = SparseStabYGeneric<BitSet, R>;

/// Y-convention sparse stabilizer simulator using `BitSet`.
pub type SparseStabYBitSet<R = PecosRng> = SparseStabYGeneric<BitSet, R>;

/// Y-convention sparse stabilizer simulator using `SortedVecSet`.
pub type SparseStabYVecSet<R = PecosRng> = SparseStabYGeneric<SortedVecSet, R>;

/// Y-convention sparse stabilizer simulator using unsorted `VecSet`.
pub type SparseStabYUnsortedVecSet<R = PecosRng> = SparseStabYGeneric<VecSet<usize>, R>;

// Constructors for default BitSet + PecosRng
impl SparseStabYGeneric<BitSet, PecosRng> {
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let rng = rand::make_rng();
        Self::with_rng(num_qubits, rng)
    }

    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let rng = PecosRng::seed_from_u64(seed);
        Self::with_rng(num_qubits, rng)
    }
}

impl SparseStabYGeneric<SortedVecSet, PecosRng> {
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let rng = rand::make_rng();
        Self::with_rng(num_qubits, rng)
    }

    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let rng = PecosRng::seed_from_u64(seed);
        Self::with_rng(num_qubits, rng)
    }
}

impl SparseStabYGeneric<VecSet<usize>, PecosRng> {
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let rng = rand::make_rng();
        Self::with_rng(num_qubits, rng)
    }

    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let rng = PecosRng::seed_from_u64(seed);
        Self::with_rng(num_qubits, rng)
    }
}

impl<S, R> SparseStabYGeneric<S, R>
where
    S: IndexSet,
    R: SeedableRng + Rng + Debug,
{
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    #[inline]
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        let mut stab = Self {
            num_qubits,
            stabs: GensGeneric::<S>::new(num_qubits),
            destabs: GensGeneric::<S>::new(num_qubits),
            rng,
            track_destab_signs: false,
        };
        stab.reset();
        stab
    }

    /// Enable tracking of destabilizer signs through Clifford gates.
    /// Required for STN-style decomposition that uses destabilizer phases.
    #[inline]
    #[must_use]
    pub fn with_destab_sign_tracking(mut self) -> Self {
        self.track_destab_signs = true;
        self
    }

    /// Whether destabilizer sign tracking is enabled.
    #[inline]
    pub fn tracks_destab_signs(&self) -> bool {
        self.track_destab_signs
    }

    #[inline]
    pub fn reset(&mut self) -> &mut Self {
        self.stabs.init_all_z();
        self.destabs.init_all_x();
        self
    }

    /// Returns generator data as sparse index vectors.
    pub fn gens_data(&self, is_stab: bool) -> crate::GensData {
        let gens = if is_stab { &self.stabs } else { &self.destabs };

        let col_x: Vec<Vec<usize>> = gens.col_x.iter().map(|s| s.iter().collect()).collect();
        let col_z: Vec<Vec<usize>> = gens.col_z.iter().map(|s| s.iter().collect()).collect();
        let row_x: Vec<Vec<usize>> = gens.row_x.iter().map(|s| s.iter().collect()).collect();
        let row_z: Vec<Vec<usize>> = gens.row_z.iter().map(|s| s.iter().collect()).collect();

        (col_x, col_z, row_x, row_z)
    }

    /// Utility that creates a string for the Pauli generators.
    /// In Y-convention, both X and Z bits set = Y directly, so we print Y (not W).
    #[inline]
    fn tableau_string(num_qubits: usize, gens: &GensGeneric<S>) -> String {
        let mut result =
            String::with_capacity(num_qubits * gens.row_x.len() + gens.row_x.len() + 2);
        for i in 0..gens.row_x.len() {
            if gens.signs_minus.contains(i) {
                result.push('-');
            } else {
                result.push('+');
            }
            if gens.signs_i.contains(i) {
                result.push('i');
            }

            for qubit in 0..num_qubits {
                let in_row_x = gens.row_x[i].contains(qubit);
                let in_row_z = gens.row_z[i].contains(qubit);

                let char = match (in_row_x, in_row_z) {
                    (false, false) => 'I',
                    (true, false) => 'X',
                    (false, true) => 'Z',
                    (true, true) => 'Y',
                };
                result.push(char);
            }
            result.push('\n');
        }

        result
    }

    #[inline]
    pub fn stab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.stabs)
    }

    #[inline]
    pub fn destab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.destabs)
    }

    #[inline]
    pub fn neg(&mut self, s: usize) {
        self.stabs.signs_minus.toggle(s);
    }

    #[inline]
    pub fn signs_minus(&self) -> &S {
        &self.stabs.signs_minus
    }

    #[inline]
    pub fn stabs(&self) -> &GensGeneric<S> {
        &self.stabs
    }

    #[inline]
    pub fn stabs_mut(&mut self) -> &mut GensGeneric<S> {
        &mut self.stabs
    }

    #[inline]
    pub fn destabs(&self) -> &GensGeneric<S> {
        &self.destabs
    }

    #[inline]
    pub fn destabs_mut(&mut self) -> &mut GensGeneric<S> {
        &mut self.destabs
    }

    #[inline]
    pub fn stabs_and_destabs_mut(&mut self) -> (&mut GensGeneric<S>, &mut GensGeneric<S>) {
        (&mut self.stabs, &mut self.destabs)
    }

    // ========================================================================
    // Measurement (Y-convention)
    //
    // In Y-convention, when we multiply two generators g_a * g_b, we need to
    // account for the implicit i factors from Y = iXZ. Each Y position in g_a
    // that overlaps with a non-identity position in g_b contributes a factor
    // of i, and vice versa. The W-convention intersection_count logic works
    // for the XZ commutation part, but we need an additional correction for
    // the Y count change.
    //
    // When multiplying generators in the Y convention:
    // - The W-convention sign logic (intersection_count of row_z with
    //   cumulative_x) handles the commutation of X and Z parts
    // - We additionally need: for each pair of generators being multiplied,
    //   count how n_Y changes and apply i^{delta_nY} correction
    //
    // For the deterministic measurement, the product g = d_1 * s_1 * d_2 * s_2 * ...
    // is computed. In Y-convention, each generator's Y-count contributes i^{n_Y}
    // implicitly. When we row-reduce, the delta_nY from each XOR gives a sign
    // correction.
    // ========================================================================

    /// Count the number of Y positions (both X and Z bits set) in a generator row.
    #[inline]
    fn count_y(row_x: &S, row_z: &S) -> usize {
        row_x.intersection_count(row_z)
    }

    #[inline]
    fn deterministic_meas(&mut self, q: usize) -> MeasurementResult {
        // Y-convention deterministic measurement.
        //
        // The formula computes the sign of the product of stabs selected by
        // destabs that have X on qubit q. Same as W-convention, plus a correction
        // for the implicit i^{n_Y} per stab from Y-convention encoding.
        //
        // The destab rows only determine WHICH stabs participate -- their phases
        // do not enter the formula. (Same structural cancellation as W-convention.)

        let mut num_minuses = self.destabs.col_x[q].intersection_count(&self.stabs.signs_minus);
        let mut num_is = self.destabs.col_x[q].intersection_count(&self.stabs.signs_i);

        // Y-convention correction: add n_Y per participating stab
        for row in self.destabs.col_x[q].iter() {
            num_is += Self::count_y(&self.stabs.row_x[row], &self.stabs.row_z[row]);
        }

        // W-convention commutation phase accumulation
        let mut cumulative_x = S::new();
        for row in self.destabs.col_x[q].iter() {
            num_minuses += self.stabs.row_z[row].intersection_count(&cumulative_x);
            cumulative_x.xor_assign(&self.stabs.row_x[row]);
        }

        // Convert i-count to sign: i^2 = -1
        if num_is & 3 != 0 {
            num_minuses += 1;
        }
        let outcome = num_minuses & 1 != 0;
        MeasurementResult {
            outcome,
            is_deterministic: true,
        }
    }

    #[allow(clippy::too_many_lines)]
    #[inline]
    fn nondeterministic_meas(&mut self, q: usize, result: bool) -> MeasurementResult {
        // Y-convention nondeterministic measurement.
        //
        // Same structural operations as W-convention (row XOR, column updates).
        // Additional sign tracking for Y-count changes:
        //
        // When XORing stab g with removed stab r, the Y-convention phase change is:
        //   i^{n_Y_r + n_Y_g_before - n_Y_g_after} * (-1)^{commutation}
        //
        // When XORing destab rows with the removed stab row, the destab Y-count
        // changes, requiring destabs.signs_i updates to maintain consistency for
        // future deterministic measurements.

        let mut anticom_stabs_col = self.stabs.col_x[q].clone();

        let mut smallest_wt = 2 * self.num_qubits + 2;
        let mut removed_id: Option<usize> = None;

        for stab_id in anticom_stabs_col.iter() {
            let weight = self.stabs.row_x[stab_id].len() + self.stabs.row_z[stab_id].len();

            if weight < smallest_wt {
                smallest_wt = weight;
                removed_id = Some(stab_id);
                if weight == 1 {
                    break;
                }
            }
        }

        let id = removed_id.expect("Critical error: removed_id was None");

        anticom_stabs_col.remove(id);
        let removed_row_x = self.stabs.row_x[id].take_clearing();
        let removed_row_z = self.stabs.row_z[id].take_clearing();

        // Y-count of the removed generator
        let removed_ny = Self::count_y(&removed_row_x, &removed_row_z);

        // Sign propagation for signs_minus
        if self.stabs.signs_minus.contains(id) {
            self.stabs.signs_minus.xor_assign(&anticom_stabs_col);
        }

        // Sign propagation for signs_i
        if self.stabs.signs_i.contains(id) {
            self.stabs.signs_i.remove(id);
            self.stabs
                .signs_i
                .xor_intersection_into(&anticom_stabs_col, &mut self.stabs.signs_minus);
            self.stabs.signs_i.xor_assign(&anticom_stabs_col);
        }

        // XOR the removed row into each anticommuting stab generator
        for g in anticom_stabs_col.iter() {
            // W-convention commutation phase
            let num_commute = removed_row_z.intersection_count(&self.stabs.row_x[g]);

            // Y-convention correction: count Y-positions before and after XOR
            let ny_g_before = Self::count_y(&self.stabs.row_x[g], &self.stabs.row_z[g]);

            // Perform the XOR
            self.stabs.row_x[g].xor_assign(&removed_row_x);
            self.stabs.row_z[g].xor_assign(&removed_row_z);

            let ny_g_after = Self::count_y(&self.stabs.row_x[g], &self.stabs.row_z[g]);

            // i-power from Y-count change: removed_ny + ny_g_before - ny_g_after
            // Plus commutation: i^{2 * num_commute} = (-1)^{num_commute}
            let y_delta = (removed_ny + ny_g_before + 4 * self.num_qubits) - ny_g_after;
            let total_sign_flips = num_commute + (y_delta >> 1);

            if total_sign_flips & 1 != 0 {
                self.stabs.signs_minus.toggle(g);
            }

            if y_delta & 1 != 0 {
                self.stabs.signs_i.toggle(g);
            }
        }

        // Column updates (structural, identical to W-convention)
        for i in removed_row_x.iter() {
            self.stabs.col_x[i].xor_assign(&anticom_stabs_col);
            self.stabs.col_x[i].remove(id);
        }

        for i in removed_row_z.iter() {
            self.stabs.col_z[i].xor_assign(&anticom_stabs_col);
            self.stabs.col_z[i].remove(id);
        }

        // Replace removed stabilizer with Z_q
        self.stabs.col_z[q].insert(id);
        self.stabs.row_z[id].insert(q);

        // Update destabilizers
        for i in self.destabs.row_x[id].iter() {
            self.destabs.col_x[i].remove(id);
        }

        for i in self.destabs.row_z[id].iter() {
            self.destabs.col_z[i].remove(id);
        }

        let mut anticom_destabs_col = self.destabs.col_x[q].clone();
        anticom_destabs_col.remove(id);

        for i in removed_row_x.iter() {
            self.destabs.col_x[i].insert(id);
            self.destabs.col_x[i].xor_assign(&anticom_destabs_col);
        }

        for i in removed_row_z.iter() {
            self.destabs.col_z[i].insert(id);
            self.destabs.col_z[i].xor_assign(&anticom_destabs_col);
        }

        // XOR destab rows (structural, same as W-convention -- destab phases not tracked)
        for row in anticom_destabs_col.iter() {
            self.destabs.row_x[row].xor_assign(&removed_row_x);
            self.destabs.row_z[row].xor_assign(&removed_row_z);
        }

        self.destabs.row_x[id] = removed_row_x;
        self.destabs.row_z[id] = removed_row_z;

        let outcome = self.apply_outcome(id, result);
        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }

    #[inline]
    pub fn mz_forced(&mut self, q: usize, forced_outcome: bool) -> MeasurementResult {
        if self.stabs.col_x[q].is_empty() {
            self.deterministic_meas(q)
        } else {
            self.nondeterministic_meas(q, forced_outcome)
        }
    }

    #[inline]
    pub fn pz_forced(&mut self, q: usize, forced_outcome: bool) -> &mut Self {
        let result = self.mz_forced(q, forced_outcome);
        if result.outcome {
            self.stabs.signs_minus.xor_assign(&self.stabs.col_z[q]);
        }
        self
    }

    #[inline]
    fn apply_outcome(&mut self, id: usize, meas_outcome: bool) -> bool {
        if meas_outcome {
            self.stabs.signs_minus.insert(id);
        } else {
            self.stabs.signs_minus.remove(id);
        }
        // Clear signs_i for the new Z_q generator (it has n_Y = 0)
        self.stabs.signs_i.remove(id);
        meas_outcome
    }
}

impl<S, R> QuantumSimulator for SparseStabYGeneric<S, R>
where
    S: IndexSet,
    R: SeedableRng + Rng + Debug,
{
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    #[inline]
    fn reset(&mut self) -> &mut Self {
        Self::reset(self)
    }
}

impl<S, R> CliffordGateable for SparseStabYGeneric<S, R>
where
    S: IndexSet,
    R: SeedableRng + Rng + Debug,
{
    /// Pauli X gate. X -> X, Z -> -Z
    /// Same as W-convention.
    #[inline]
    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.signs_minus.xor_assign(&self.stabs.col_z[qu]);
            if self.track_destab_signs {
                self.destabs.signs_minus.xor_assign(&self.destabs.col_z[qu]);
            }
        }
        self
    }

    /// Pauli Y gate. X -> -X, Z -> -Z
    /// Same as W-convention.
    #[inline]
    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.col_x[qu]
                .xor_symmetric_difference_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            if self.track_destab_signs {
                self.destabs.col_x[qu].xor_symmetric_difference_into(
                    &self.destabs.col_z[qu],
                    &mut self.destabs.signs_minus,
                );
            }
        }
        self
    }

    /// Pauli Z gate. X -> -X, Z -> Z
    /// Same as W-convention.
    #[inline]
    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.stabs
                .signs_minus
                .xor_assign(&self.stabs.col_x[q.index()]);
            if self.track_destab_signs {
                self.destabs
                    .signs_minus
                    .xor_assign(&self.destabs.col_x[q.index()]);
            }
        }
        self
    }

    /// Sqrt of Z gate (Y-convention).
    ///     X -> Y (= iXZ, stored as both bits set with no sign change)
    ///     Z -> Z
    ///     Y -> -X (Y = iXZ, SZ(Y) = SZ(iXZ) = i * Y * Z = i * iXZ * Z = i * iX = -X)
    ///
    /// In Y-convention: `signs_minus ^= col_x[q] AND col_z[q]`
    /// (no `signs_i` update needed!)
    #[inline]
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();

            // Y-convention sign update: toggle minus for generators that have Y on this qubit
            // (both X and Z bits set). Y -> -X means the sign flips.
            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            if self.track_destab_signs {
                self.destabs.col_x[qu]
                    .xor_intersection_into(&self.destabs.col_z[qu], &mut self.destabs.signs_minus);
            }

            // Data update: same as W-convention (X bit implies Z bit gets toggled)
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_z[qu].xor_assign(&g.col_x[qu]);

                for i in g.col_x[qu].iter() {
                    g.row_z[i].toggle(qu);
                }
            }
        }
        self
    }

    /// Hadamard gate. X -> Z, Z -> X
    /// Y-convention sign: `signs_minus` ^= `col_x`[q] AND `col_z`[q] (same as W-convention)
    #[inline]
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();

            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            if self.track_destab_signs {
                self.destabs.col_x[qu]
                    .xor_intersection_into(&self.destabs.col_z[qu], &mut self.destabs.signs_minus);
            }

            for g in [&mut self.stabs, &mut self.destabs] {
                for i in g.col_x[qu].iter() {
                    if !g.col_z[qu].contains(i) {
                        g.row_x[i].remove(qu);
                        g.row_z[i].insert(qu);
                    }
                }

                for i in g.col_z[qu].iter() {
                    if !g.col_x[qu].contains(i) {
                        g.row_z[i].remove(qu);
                        g.row_x[i].insert(qu);
                    }
                }

                mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
            }
        }
        self
    }

    /// `SZdg` (Y-convention): X→-Y, Z→+Z. Y→+X.
    /// Sign: toggle minus for `col_x` \ `col_z` (only X generators, not Y).
    #[inline]
    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.signs_minus.xor_assign(&self.stabs.col_x[qu]);
            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            if self.track_destab_signs {
                self.destabs.signs_minus.xor_assign(&self.destabs.col_x[qu]);
                self.destabs.col_x[qu]
                    .xor_intersection_into(&self.destabs.col_z[qu], &mut self.destabs.signs_minus);
            }
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_z[qu].xor_assign(&g.col_x[qu]);
                for i in g.col_x[qu].iter() {
                    g.row_z[i].toggle(qu);
                }
            }
        }
        self
    }

    /// SX (Y-convention): X→+X, Z→-Y. Y→+Z.
    /// Sign: toggle minus for `col_z` \ `col_x` (only Z generators, not Y).
    #[inline]
    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.signs_minus.xor_assign(&self.stabs.col_z[qu]);
            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            if self.track_destab_signs {
                self.destabs.signs_minus.xor_assign(&self.destabs.col_z[qu]);
                self.destabs.col_x[qu]
                    .xor_intersection_into(&self.destabs.col_z[qu], &mut self.destabs.signs_minus);
            }
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_x[qu].xor_assign(&g.col_z[qu]);
                for i in g.col_z[qu].iter() {
                    g.row_x[i].toggle(qu);
                }
            }
        }
        self
    }

    /// `SXdg` (Y-convention): X→+X, Z→+Y. Y→-Z.
    /// Sign: toggle minus for `col_x` ∩ `col_z` (only Y generators).
    #[inline]
    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            if self.track_destab_signs {
                self.destabs.col_x[qu]
                    .xor_intersection_into(&self.destabs.col_z[qu], &mut self.destabs.signs_minus);
            }
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_x[qu].xor_assign(&g.col_z[qu]);
                for i in g.col_z[qu].iter() {
                    g.row_x[i].toggle(qu);
                }
            }
        }
        self
    }

    /// SY (Y-convention): X→-Z, Z→+X. Y→+Y.
    /// Sign: toggle minus for `col_x` \ `col_z` (only X generators).
    #[inline]
    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.signs_minus.xor_assign(&self.stabs.col_x[qu]);
            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            if self.track_destab_signs {
                self.destabs.signs_minus.xor_assign(&self.destabs.col_x[qu]);
                self.destabs.col_x[qu]
                    .xor_intersection_into(&self.destabs.col_z[qu], &mut self.destabs.signs_minus);
            }
            for g in [&mut self.stabs, &mut self.destabs] {
                for i in g.col_x[qu].iter() {
                    if !g.col_z[qu].contains(i) {
                        g.row_x[i].remove(qu);
                        g.row_z[i].insert(qu);
                    }
                }
                for i in g.col_z[qu].iter() {
                    if !g.col_x[qu].contains(i) {
                        g.row_z[i].remove(qu);
                        g.row_x[i].insert(qu);
                    }
                }
                mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
            }
        }
        self
    }

    /// `SYdg` (Y-convention): X→+Z, Z→-X. Y→+Y.
    /// Sign: toggle minus for `col_z` \ `col_x` (only Z generators).
    #[inline]
    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.signs_minus.xor_assign(&self.stabs.col_z[qu]);
            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            if self.track_destab_signs {
                self.destabs.signs_minus.xor_assign(&self.destabs.col_z[qu]);
                self.destabs.col_x[qu]
                    .xor_intersection_into(&self.destabs.col_z[qu], &mut self.destabs.signs_minus);
            }
            for g in [&mut self.stabs, &mut self.destabs] {
                for i in g.col_x[qu].iter() {
                    if !g.col_z[qu].contains(i) {
                        g.row_x[i].remove(qu);
                        g.row_z[i].insert(qu);
                    }
                }
                for i in g.col_z[qu].iter() {
                    if !g.col_x[qu].contains(i) {
                        g.row_z[i].remove(qu);
                        g.row_x[i].insert(qu);
                    }
                }
                mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
            }
        }
        self
    }

    /// H2 (Y-convention): X→-Z, Z→-X. Y→-Y.
    /// Sign: toggle minus for all non-identity generators.
    #[inline]
    fn h2(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.col_x[qu]
                .xor_symmetric_difference_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            for g in [&mut self.stabs, &mut self.destabs] {
                for i in g.col_x[qu].iter() {
                    if !g.col_z[qu].contains(i) {
                        g.row_x[i].remove(qu);
                        g.row_z[i].insert(qu);
                    }
                }
                for i in g.col_z[qu].iter() {
                    if !g.col_x[qu].contains(i) {
                        g.row_z[i].remove(qu);
                        g.row_x[i].insert(qu);
                    }
                }
                mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
            }
        }
        self
    }

    /// H3 (Y-convention): X→+Y, Z→-Z. Y→+X.
    /// Sign: toggle minus for `col_z` \ `col_x` (only Z generators).
    #[inline]
    fn h3(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.signs_minus.xor_assign(&self.stabs.col_z[qu]);
            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_z[qu].xor_assign(&g.col_x[qu]);
                for i in g.col_x[qu].iter() {
                    g.row_z[i].toggle(qu);
                }
            }
        }
        self
    }

    /// H4 (Y-convention): X→-Y, Z→-Z. Y→-X.
    /// Sign: toggle minus for all non-identity generators.
    #[inline]
    fn h4(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.col_x[qu]
                .xor_symmetric_difference_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_z[qu].xor_assign(&g.col_x[qu]);
                for i in g.col_x[qu].iter() {
                    g.row_z[i].toggle(qu);
                }
            }
        }
        self
    }

    /// H5 (Y-convention): X→-X, Z→+Y. Y→+Z.
    /// Sign: toggle minus for `col_x` \ `col_z` (only X generators).
    #[inline]
    fn h5(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.signs_minus.xor_assign(&self.stabs.col_x[qu]);
            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_x[qu].xor_assign(&g.col_z[qu]);
                for i in g.col_z[qu].iter() {
                    g.row_x[i].toggle(qu);
                }
            }
        }
        self
    }

    /// H6 (Y-convention): X→-X, Z→-Y. Y→-Z.
    /// Sign: toggle minus for all non-identity generators.
    #[inline]
    fn h6(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.col_x[qu]
                .xor_symmetric_difference_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_x[qu].xor_assign(&g.col_z[qu]);
                for i in g.col_z[qu].iter() {
                    g.row_x[i].toggle(qu);
                }
            }
        }
        self
    }

    /// F (Y-convention): X→+Y, Z→+X. Y→+Z.
    /// Sign: none (all images positive).
    #[inline]
    fn f(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            // Data: col_z ^= col_x, then swap col_x ↔ col_z.
            // Net effect: new_x = old_x ⊕ old_z, new_z = old_x.
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_z[qu].xor_assign(&g.col_x[qu]);
                for i in g.col_x[qu].iter() {
                    g.row_z[i].toggle(qu);
                }
                for i in g.col_x[qu].iter() {
                    if !g.col_z[qu].contains(i) {
                        g.row_x[i].remove(qu);
                        g.row_z[i].insert(qu);
                    }
                }
                for i in g.col_z[qu].iter() {
                    if !g.col_x[qu].contains(i) {
                        g.row_z[i].remove(qu);
                        g.row_x[i].insert(qu);
                    }
                }
                mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
            }
        }
        self
    }

    /// Fdg (Y-convention): X→+Z, Z→+Y. Y→+X.
    /// Sign: none (all images positive).
    #[inline]
    fn fdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            // Data: col_x ^= col_z, then swap col_x ↔ col_z.
            // Net effect: new_x = old_z, new_z = old_x ⊕ old_z.
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_x[qu].xor_assign(&g.col_z[qu]);
                for i in g.col_z[qu].iter() {
                    g.row_x[i].toggle(qu);
                }
                for i in g.col_x[qu].iter() {
                    if !g.col_z[qu].contains(i) {
                        g.row_x[i].remove(qu);
                        g.row_z[i].insert(qu);
                    }
                }
                for i in g.col_z[qu].iter() {
                    if !g.col_x[qu].contains(i) {
                        g.row_z[i].remove(qu);
                        g.row_x[i].insert(qu);
                    }
                }
                mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
            }
        }
        self
    }

    /// F2 (Y-convention): X→-Z, Z→+Y. Y→-X.
    /// Sign: toggle minus for `col_x` (= {X,Y}).
    #[inline]
    fn f2(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.signs_minus.xor_assign(&self.stabs.col_x[qu]);
            // Data: col_x ^= col_z, then swap.
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_x[qu].xor_assign(&g.col_z[qu]);
                for i in g.col_z[qu].iter() {
                    g.row_x[i].toggle(qu);
                }
                for i in g.col_x[qu].iter() {
                    if !g.col_z[qu].contains(i) {
                        g.row_x[i].remove(qu);
                        g.row_z[i].insert(qu);
                    }
                }
                for i in g.col_z[qu].iter() {
                    if !g.col_x[qu].contains(i) {
                        g.row_z[i].remove(qu);
                        g.row_x[i].insert(qu);
                    }
                }
                mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
            }
        }
        self
    }

    /// F2dg (Y-convention): X→-Y, Z→-X. Y→+Z.
    /// Sign: toggle minus for `col_x` ⊕ `col_z` (= {X,Z}).
    #[inline]
    fn f2dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.col_x[qu]
                .xor_symmetric_difference_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            // Data: col_z ^= col_x, then swap.
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_z[qu].xor_assign(&g.col_x[qu]);
                for i in g.col_x[qu].iter() {
                    g.row_z[i].toggle(qu);
                }
                for i in g.col_x[qu].iter() {
                    if !g.col_z[qu].contains(i) {
                        g.row_x[i].remove(qu);
                        g.row_z[i].insert(qu);
                    }
                }
                for i in g.col_z[qu].iter() {
                    if !g.col_x[qu].contains(i) {
                        g.row_z[i].remove(qu);
                        g.row_x[i].insert(qu);
                    }
                }
                mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
            }
        }
        self
    }

    /// F3 (Y-convention): X→+Y, Z→-X. Y→-Z.
    /// Sign: toggle minus for `col_z` (= {Z,Y}).
    #[inline]
    fn f3(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.signs_minus.xor_assign(&self.stabs.col_z[qu]);
            // Data: col_z ^= col_x, then swap.
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_z[qu].xor_assign(&g.col_x[qu]);
                for i in g.col_x[qu].iter() {
                    g.row_z[i].toggle(qu);
                }
                for i in g.col_x[qu].iter() {
                    if !g.col_z[qu].contains(i) {
                        g.row_x[i].remove(qu);
                        g.row_z[i].insert(qu);
                    }
                }
                for i in g.col_z[qu].iter() {
                    if !g.col_x[qu].contains(i) {
                        g.row_z[i].remove(qu);
                        g.row_x[i].insert(qu);
                    }
                }
                mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
            }
        }
        self
    }

    /// F3dg (Y-convention): X→-Z, Z→-Y. Y→+X.
    /// Sign: toggle minus for `col_x` ⊕ `col_z` (= {X,Z}).
    #[inline]
    fn f3dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.col_x[qu]
                .xor_symmetric_difference_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
            // Data: col_x ^= col_z, then swap.
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_x[qu].xor_assign(&g.col_z[qu]);
                for i in g.col_z[qu].iter() {
                    g.row_x[i].toggle(qu);
                }
                for i in g.col_x[qu].iter() {
                    if !g.col_z[qu].contains(i) {
                        g.row_x[i].remove(qu);
                        g.row_z[i].insert(qu);
                    }
                }
                for i in g.col_z[qu].iter() {
                    if !g.col_x[qu].contains(i) {
                        g.row_z[i].remove(qu);
                        g.row_x[i].insert(qu);
                    }
                }
                mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
            }
        }
        self
    }

    /// F4 (Y-convention): X→+Z, Z→-Y. Y→-X.
    /// Sign: toggle minus for `col_z` (= {Z,Y}).
    #[inline]
    fn f4(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.signs_minus.xor_assign(&self.stabs.col_z[qu]);
            // Data: col_x ^= col_z, then swap.
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_x[qu].xor_assign(&g.col_z[qu]);
                for i in g.col_z[qu].iter() {
                    g.row_x[i].toggle(qu);
                }
                for i in g.col_x[qu].iter() {
                    if !g.col_z[qu].contains(i) {
                        g.row_x[i].remove(qu);
                        g.row_z[i].insert(qu);
                    }
                }
                for i in g.col_z[qu].iter() {
                    if !g.col_x[qu].contains(i) {
                        g.row_z[i].remove(qu);
                        g.row_x[i].insert(qu);
                    }
                }
                mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
            }
        }
        self
    }

    /// F4dg (Y-convention): X→-Y, Z→+X. Y→-Z.
    /// Sign: toggle minus for `col_x` (= {X,Y}).
    #[inline]
    fn f4dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.signs_minus.xor_assign(&self.stabs.col_x[qu]);
            // Data: col_z ^= col_x, then swap.
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_z[qu].xor_assign(&g.col_x[qu]);
                for i in g.col_x[qu].iter() {
                    g.row_z[i].toggle(qu);
                }
                for i in g.col_x[qu].iter() {
                    if !g.col_z[qu].contains(i) {
                        g.row_x[i].remove(qu);
                        g.row_z[i].insert(qu);
                    }
                }
                for i in g.col_z[qu].iter() {
                    if !g.col_x[qu].contains(i) {
                        g.row_z[i].remove(qu);
                        g.row_x[i].insert(qu);
                    }
                }
                mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
            }
        }
        self
    }

    /// CX gate (Y-convention).
    ///
    /// Data update: same as W-convention (XI->XX, IZ->ZZ, ZI->ZI, IX->IX)
    /// Sign update: `signs_minus ^= col_x[q1] AND col_z[q2] AND NOT(col_z[q1] XOR col_x[q2])`
    ///
    /// This toggles signs for generators where:
    /// - q1 has X (either X or Y) and q2 has Z (either Z or Y)
    /// - AND q1's Z matches q2's X (both absent or both present)
    ///   i.e., (X,Z) or (Y,Y) on (q1,q2) -- exactly the cases where `n_Y` changes by +/-2.
    #[inline]
    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(control, target) in pairs {
            let q1 = control.index();
            let q2 = target.index();
            debug_assert_ne!(q1, q2, "CX requires distinct qubits");

            // Y-convention sign update
            // Toggle signs_minus where: col_x[q1] AND col_z[q2] AND NOT(col_z[q1] XOR col_x[q2])
            // = col_x[q1] AND col_z[q2] AND ((col_z[q1] AND col_x[q2]) OR (NOT col_z[q1] AND NOT col_x[q2]))
            // = (col_x[q1] AND col_z[q2] AND col_z[q1] AND col_x[q2]) OR (col_x[q1] AND col_z[q2] AND NOT col_z[q1] AND NOT col_x[q2])
            //
            // Case 1: x1 AND z2 AND z1 AND x2 -- both qubits have Y (both bits set)
            // Case 2: x1 AND z2 AND NOT z1 AND NOT x2 -- q1 has pure X, q2 has pure Z
            for g in self.stabs.col_x[q1].iter() {
                if !self.stabs.col_z[q2].contains(g) {
                    continue;
                }
                let has_z1 = self.stabs.col_z[q1].contains(g);
                let has_x2 = self.stabs.col_x[q2].contains(g);
                if has_z1 == has_x2 {
                    self.stabs.signs_minus.toggle(g);
                }
            }
            if self.track_destab_signs {
                for g in self.destabs.col_x[q1].iter() {
                    if !self.destabs.col_z[q2].contains(g) {
                        continue;
                    }
                    let has_z1 = self.destabs.col_z[q1].contains(g);
                    let has_x2 = self.destabs.col_x[q2].contains(g);
                    if has_z1 == has_x2 {
                        self.destabs.signs_minus.toggle(g);
                    }
                }
            }

            // Data update: identical to W-convention
            for g in &mut [&mut self.stabs, &mut self.destabs] {
                unsafe {
                    let col_x_q1 = g.col_x.get_unchecked(q1);
                    for i in col_x_q1.iter() {
                        g.row_x.get_unchecked_mut(i).toggle(q2);
                    }
                    let col_x_q1 = std::ptr::from_ref::<S>(g.col_x.get_unchecked(q1));
                    let col_x_q2 = g.col_x.get_unchecked_mut(q2);
                    col_x_q2.xor_assign(&*col_x_q1);

                    let col_z_q2 = g.col_z.get_unchecked(q2);
                    for i in col_z_q2.iter() {
                        g.row_z.get_unchecked_mut(i).toggle(q1);
                    }
                    let col_z_q2 = std::ptr::from_ref::<S>(g.col_z.get_unchecked(q2));
                    let col_z_q1 = g.col_z.get_unchecked_mut(q1);
                    col_z_q1.xor_assign(&*col_z_q2);
                }
            }
        }
        self
    }

    /// Square root of XX gate (Y-convention).
    ///
    /// Generators with odd Z-count on {q1,q2}: toggle X on both qubits.
    /// Sign update: toggle `signs_minus` when the odd-Z qubit had Y (both x,z set),
    /// because removing Y's implicit i requires a stored phase correction.
    #[inline]
    fn sxx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();
            debug_assert_ne!(q1, q2, "SXX requires distinct qubits");

            // Sign update: Q -> i*Q*XX. Per-qubit phase from
            // right-multiplying by X: Z*X=iY (c=+i), Y*X=-iZ (c=-i).
            // For odd-Z generators (z=1 at one qubit), total = i*c_q:
            //   z=1 qubit is Z (x=0): i*(+i) = -1 -> toggle signs_minus
            //   z=1 qubit is Y (x=1): i*(-i) = +1 -> no toggle
            for g in self.stabs.col_z[q1].iter() {
                if !self.stabs.col_z[q2].contains(g) && !self.stabs.col_x[q1].contains(g) {
                    self.stabs.signs_minus.toggle(g);
                }
            }
            for g in self.stabs.col_z[q2].iter() {
                if !self.stabs.col_z[q1].contains(g) && !self.stabs.col_x[q2].contains(g) {
                    self.stabs.signs_minus.toggle(g);
                }
            }
            if self.track_destab_signs {
                for g in self.destabs.col_z[q1].iter() {
                    if !self.destabs.col_z[q2].contains(g) && !self.destabs.col_x[q1].contains(g) {
                        self.destabs.signs_minus.toggle(g);
                    }
                }
                for g in self.destabs.col_z[q2].iter() {
                    if !self.destabs.col_z[q1].contains(g) && !self.destabs.col_x[q2].contains(g) {
                        self.destabs.signs_minus.toggle(g);
                    }
                }
            }

            // Pauli update (both stabs and destabs): toggle X on q1,q2 for odd-Z generators.
            for tab in [&mut self.stabs, &mut self.destabs] {
                unsafe {
                    let col_z_q1 = std::ptr::from_ref::<S>(tab.col_z.get_unchecked(q1));
                    let col_z_q2 = std::ptr::from_ref::<S>(tab.col_z.get_unchecked(q2));
                    let col_x_q1 = tab.col_x.get_unchecked_mut(q1);
                    let old_col_x_q1 = col_x_q1.clone();
                    col_x_q1.xor_assign(&*col_z_q1);
                    col_x_q1.xor_assign(&*col_z_q2);
                    for i in old_col_x_q1.iter() {
                        if !tab.col_x.get_unchecked(q1).contains(i) {
                            tab.row_x.get_unchecked_mut(i).remove(q1);
                        }
                    }
                    for i in tab.col_x.get_unchecked(q1).iter() {
                        if !old_col_x_q1.contains(i) {
                            tab.row_x.get_unchecked_mut(i).insert(q1);
                        }
                    }

                    let col_z_q1 = std::ptr::from_ref::<S>(tab.col_z.get_unchecked(q1));
                    let col_z_q2 = std::ptr::from_ref::<S>(tab.col_z.get_unchecked(q2));
                    let col_x_q2 = tab.col_x.get_unchecked_mut(q2);
                    let old_col_x_q2 = col_x_q2.clone();
                    col_x_q2.xor_assign(&*col_z_q1);
                    col_x_q2.xor_assign(&*col_z_q2);
                    for i in old_col_x_q2.iter() {
                        if !tab.col_x.get_unchecked(q2).contains(i) {
                            tab.row_x.get_unchecked_mut(i).remove(q2);
                        }
                    }
                    for i in tab.col_x.get_unchecked(q2).iter() {
                        if !old_col_x_q2.contains(i) {
                            tab.row_x.get_unchecked_mut(i).insert(q2);
                        }
                    }
                }
            }
        }
        self
    }

    /// Adjoint of square root of XX gate. `SXXdg` = X(q1).X(q2).SXX
    #[inline]
    fn sxxdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let q1s: Vec<QubitId> = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: Vec<QubitId> = pairs.iter().map(|&(_, q2)| q2).collect();
        self.x(&q1s).x(&q2s).sxx(pairs)
    }

    /// Square root of ZZ gate (Y-convention).
    ///
    /// Generators with odd X-count on {q1,q2}: toggle Z on both qubits.
    /// Sign update: toggle `signs_minus` when the odd-X qubit had Y (both x,z set).
    #[inline]
    fn szz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();
            debug_assert_ne!(q1, q2, "SZZ requires distinct qubits");

            // Sign update (stabs only): Q -> i*Q*ZZ. Per-qubit phase from
            // right-multiplying by Z: X*Z=-iY (c=-i), Y*Z=iX (c=+i).
            // For odd-X generators (x=1 at one qubit), total = i*c_q:
            //   x=1 qubit is X (z=0): i*(-i) = +1 -> no toggle
            //   x=1 qubit is Y (z=1): i*(+i) = -1 -> toggle signs_minus
            for g in self.stabs.col_x[q1].iter() {
                if !self.stabs.col_x[q2].contains(g) && self.stabs.col_z[q1].contains(g) {
                    self.stabs.signs_minus.toggle(g);
                }
            }
            for g in self.stabs.col_x[q2].iter() {
                if !self.stabs.col_x[q1].contains(g) && self.stabs.col_z[q2].contains(g) {
                    self.stabs.signs_minus.toggle(g);
                }
            }

            for tab in [&mut self.stabs, &mut self.destabs] {
                unsafe {
                    let col_x_q1 = std::ptr::from_ref::<S>(tab.col_x.get_unchecked(q1));
                    let col_x_q2 = std::ptr::from_ref::<S>(tab.col_x.get_unchecked(q2));
                    let col_z_q1 = tab.col_z.get_unchecked_mut(q1);
                    let old_col_z_q1 = col_z_q1.clone();
                    col_z_q1.xor_assign(&*col_x_q1);
                    col_z_q1.xor_assign(&*col_x_q2);
                    for i in old_col_z_q1.iter() {
                        if !tab.col_z.get_unchecked(q1).contains(i) {
                            tab.row_z.get_unchecked_mut(i).remove(q1);
                        }
                    }
                    for i in tab.col_z.get_unchecked(q1).iter() {
                        if !old_col_z_q1.contains(i) {
                            tab.row_z.get_unchecked_mut(i).insert(q1);
                        }
                    }

                    let col_x_q1 = std::ptr::from_ref::<S>(tab.col_x.get_unchecked(q1));
                    let col_x_q2 = std::ptr::from_ref::<S>(tab.col_x.get_unchecked(q2));
                    let col_z_q2 = tab.col_z.get_unchecked_mut(q2);
                    let old_col_z_q2 = col_z_q2.clone();
                    col_z_q2.xor_assign(&*col_x_q1);
                    col_z_q2.xor_assign(&*col_x_q2);
                    for i in old_col_z_q2.iter() {
                        if !tab.col_z.get_unchecked(q2).contains(i) {
                            tab.row_z.get_unchecked_mut(i).remove(q2);
                        }
                    }
                    for i in tab.col_z.get_unchecked(q2).iter() {
                        if !old_col_z_q2.contains(i) {
                            tab.row_z.get_unchecked_mut(i).insert(q2);
                        }
                    }
                }
            }
        }
        self
    }

    /// Adjoint of square root of ZZ gate. `SZZdg` = Z(q1).Z(q2).SZZ
    #[inline]
    fn szzdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let q1s: Vec<QubitId> = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: Vec<QubitId> = pairs.iter().map(|&(_, q2)| q2).collect();
        self.z(&q1s).z(&q2s).szz(pairs)
    }

    /// Square root of YY gate (Y-convention).
    ///
    /// Generators where odd number of {q1,q2} anticommute with Y: toggle both X,Z on both qubits.
    /// Sign update: toggle `signs_minus` when the non-anticommuting qubit had Y (both x,z set).
    #[inline]
    fn syy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();
            debug_assert_ne!(q1, q2, "SYY requires distinct qubits");

            // Sign update (stabs only): Q -> i*Q*YY. Per-qubit phase from
            // right-multiplying by Y: X*Y=iZ (c=+i), Z*Y=-iX (c=-i).
            // For the anticommuting qubit (x!=z), total = i*c:
            //   X (x=1,z=0): i*(+i) = -1 -> toggle signs_minus
            //   Z (x=0,z=1): i*(-i) = +1 -> no toggle
            // Commuting qubit (x=z) always has c=+1 (I*Y=Y or Y*Y=I).
            //
            // Iterate over generators with X at q1 (x1=1, z1=0) where q2 commutes (x2=z2).
            for g in self.stabs.col_x[q1].iter() {
                if self.stabs.col_z[q1].contains(g) {
                    continue;
                } // skip Y, need X (x=1,z=0)
                let x2 = self.stabs.col_x[q2].contains(g);
                let z2 = self.stabs.col_z[q2].contains(g);
                if x2 == z2 {
                    // q2 commutes with Y -> generator is affected, toggle
                    self.stabs.signs_minus.toggle(g);
                }
            }
            // Iterate over generators with X at q2 (x2=1, z2=0) where q1 commutes (x1=z1).
            for g in self.stabs.col_x[q2].iter() {
                if self.stabs.col_z[q2].contains(g) {
                    continue;
                }
                let x1 = self.stabs.col_x[q1].contains(g);
                let z1 = self.stabs.col_z[q1].contains(g);
                if x1 == z1 {
                    self.stabs.signs_minus.toggle(g);
                }
            }
            if self.track_destab_signs {
                for g in self.destabs.col_x[q1].iter() {
                    if self.destabs.col_z[q1].contains(g) {
                        continue;
                    }
                    let x2 = self.destabs.col_x[q2].contains(g);
                    let z2 = self.destabs.col_z[q2].contains(g);
                    if x2 == z2 {
                        self.destabs.signs_minus.toggle(g);
                    }
                }
                for g in self.destabs.col_x[q2].iter() {
                    if self.destabs.col_z[q2].contains(g) {
                        continue;
                    }
                    let x1 = self.destabs.col_x[q1].contains(g);
                    let z1 = self.destabs.col_z[q1].contains(g);
                    if x1 == z1 {
                        self.destabs.signs_minus.toggle(g);
                    }
                }
            }

            for tab in [&mut self.stabs, &mut self.destabs] {
                unsafe {
                    // Compute affected set: generators where (x1^z1) XOR (x2^z2) = 1
                    let mut anti_y_q1 = tab.col_x.get_unchecked(q1).clone();
                    anti_y_q1.xor_assign(tab.col_z.get_unchecked(q1));
                    let mut anti_y_q2 = tab.col_x.get_unchecked(q2).clone();
                    anti_y_q2.xor_assign(tab.col_z.get_unchecked(q2));
                    let mut affected = anti_y_q1;
                    affected.xor_assign(&anti_y_q2);

                    // Toggle X bits at q1 and q2
                    let old_col_x_q1 = tab.col_x.get_unchecked(q1).clone();
                    tab.col_x.get_unchecked_mut(q1).xor_assign(&affected);
                    for i in old_col_x_q1.iter() {
                        if !tab.col_x.get_unchecked(q1).contains(i) {
                            tab.row_x.get_unchecked_mut(i).remove(q1);
                        }
                    }
                    for i in tab.col_x.get_unchecked(q1).iter() {
                        if !old_col_x_q1.contains(i) {
                            tab.row_x.get_unchecked_mut(i).insert(q1);
                        }
                    }

                    let old_col_x_q2 = tab.col_x.get_unchecked(q2).clone();
                    tab.col_x.get_unchecked_mut(q2).xor_assign(&affected);
                    for i in old_col_x_q2.iter() {
                        if !tab.col_x.get_unchecked(q2).contains(i) {
                            tab.row_x.get_unchecked_mut(i).remove(q2);
                        }
                    }
                    for i in tab.col_x.get_unchecked(q2).iter() {
                        if !old_col_x_q2.contains(i) {
                            tab.row_x.get_unchecked_mut(i).insert(q2);
                        }
                    }

                    // Toggle Z bits at q1 and q2
                    let old_col_z_q1 = tab.col_z.get_unchecked(q1).clone();
                    tab.col_z.get_unchecked_mut(q1).xor_assign(&affected);
                    for i in old_col_z_q1.iter() {
                        if !tab.col_z.get_unchecked(q1).contains(i) {
                            tab.row_z.get_unchecked_mut(i).remove(q1);
                        }
                    }
                    for i in tab.col_z.get_unchecked(q1).iter() {
                        if !old_col_z_q1.contains(i) {
                            tab.row_z.get_unchecked_mut(i).insert(q1);
                        }
                    }

                    let old_col_z_q2 = tab.col_z.get_unchecked(q2).clone();
                    tab.col_z.get_unchecked_mut(q2).xor_assign(&affected);
                    for i in old_col_z_q2.iter() {
                        if !tab.col_z.get_unchecked(q2).contains(i) {
                            tab.row_z.get_unchecked_mut(i).remove(q2);
                        }
                    }
                    for i in tab.col_z.get_unchecked(q2).iter() {
                        if !old_col_z_q2.contains(i) {
                            tab.row_z.get_unchecked_mut(i).insert(q2);
                        }
                    }
                }
            }
        }
        self
    }

    /// Adjoint of square root of YY gate. `SYYdg` = Y(q1).Y(q2).SYY
    #[inline]
    fn syydg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let q1s: Vec<QubitId> = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: Vec<QubitId> = pairs.iter().map(|&(_, q2)| q2).collect();
        self.y(&q1s).y(&q2s).syy(pairs)
    }

    /// Measures qubits in the Z basis.
    #[inline]
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let mut results = Vec::with_capacity(qubits.len());

        for &q in qubits {
            let qu = q.index();
            let deterministic = self.stabs.col_x[qu].is_empty();

            let result = if deterministic {
                self.deterministic_meas(qu)
            } else {
                let outcome = self.rng.coin_flip();
                self.nondeterministic_meas(qu, outcome)
            };
            results.push(result);
        }

        results
    }
}

impl<S, R> RngManageable for SparseStabYGeneric<S, R>
where
    S: IndexSet,
    R: SeedableRng + Rng + Debug,
{
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

use crate::stabilizer_tableau::StabilizerTableauSimulator;

impl<S, R> StabilizerTableauSimulator for SparseStabYGeneric<S, R>
where
    S: IndexSet,
    R: SeedableRng + Rng + Debug,
{
    fn stab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.stabs)
    }

    fn destab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.destabs)
    }
}

// ForcedMeasurement and StabilizerSimulator implementations
use crate::stabilizer_test_utils::{ForcedMeasurement, StabilizerSimulator};

impl<S, R> ForcedMeasurement for SparseStabYGeneric<S, R>
where
    S: IndexSet,
    R: SeedableRng + Rng + Debug,
{
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        SparseStabYGeneric::mz_forced(self, qubit, forced_outcome)
    }
}

impl StabilizerSimulator for SparseStabYGeneric<BitSet, PecosRng> {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::QubitId;
    use pecos_core::clifford::Clifford;

    fn q(n: usize) -> [QubitId; 1] {
        [QubitId(n)]
    }

    fn q2(a: usize, b: usize) -> [(QubitId, QubitId); 1] {
        [(QubitId(a), QubitId(b))]
    }

    /// Test that the initial state is correct: stabs = Z per qubit, destabs = X per qubit.
    #[test]
    fn test_initial_state() {
        let state = SparseStabY::new(2);
        assert_eq!(state.stab_tableau(), "+ZI\n+IZ\n");
        assert_eq!(state.destab_tableau(), "+XI\n+IX\n");
    }

    /// Test basic Pauli gates on single-qubit states.
    #[test]
    fn test_pauli_gates() {
        // X on |0> -> |1>: stab Z -> -Z
        let mut state = SparseStabY::new(1);
        state.x(&q(0));
        assert_eq!(state.stab_tableau(), "-Z\n");

        // Z on |+> -> |->: stab X -> -X
        let mut state = SparseStabY::new(1);
        state.h(&q(0));
        state.z(&q(0));
        assert_eq!(state.stab_tableau(), "-X\n");
    }

    /// Test H gate: Z -> X, X -> Z
    #[test]
    fn test_h_gate() {
        let mut state = SparseStabY::new(1);
        // |0> has stab +Z. H|0> = |+> has stab +X.
        state.h(&q(0));
        assert_eq!(state.stab_tableau(), "+X\n");
    }

    /// Test SZ gate (Y-convention): X -> Y (both bits, no i prefix), Y -> -X
    #[test]
    fn test_sz_gate() {
        // X -> Y: stab X, apply SZ, should get +Y (both bits, no i)
        let mut state = SparseStabY::new(1);
        state.h(&q(0)); // stab = +X
        state.sz(&q(0)); // stab = +Y
        assert_eq!(state.stab_tableau(), "+Y\n");

        // Z -> Z: stab Z, apply SZ, should get +Z
        let mut state = SparseStabY::new(1);
        state.sz(&q(0)); // stab = +Z
        assert_eq!(state.stab_tableau(), "+Z\n");

        // Y -> -X
        let mut state = SparseStabY::new(1);
        state.h(&q(0)); // +X
        state.sz(&q(0)); // +Y
        state.sz(&q(0)); // -X
        assert_eq!(state.stab_tableau(), "-X\n");
    }

    /// Test CX gate creates Bell state correctly.
    #[test]
    fn test_cx_bell_state() {
        let mut state = SparseStabY::new(2);
        state.h(&q(0)).cx(&q2(0, 1));
        // Bell state |00>+|11>: stabs should be +XX and +ZZ
        let tableau = state.stab_tableau();
        assert!(tableau.contains("+XX"), "Expected +XX in {tableau}");
        assert!(tableau.contains("+ZZ"), "Expected +ZZ in {tableau}");
    }

    /// Test that measurement of |0> gives deterministic 0.
    #[test]
    fn test_deterministic_meas_z0() {
        let mut state = SparseStabY::new(1);
        let r = state.mz(&q(0)).into_iter().next().unwrap();
        assert!(r.is_deterministic);
        assert!(!r.outcome); // |0> -> outcome 0
    }

    /// Test that measurement of |1> gives deterministic 1.
    #[test]
    fn test_deterministic_meas_z1() {
        let mut state = SparseStabY::new(1);
        state.x(&q(0)); // |1>: stab = -Z
        let r = state.mz(&q(0)).into_iter().next().unwrap();
        assert!(r.is_deterministic);
        assert!(r.outcome); // |1> -> outcome 1
    }

    /// Test that measurement of |+> is nondeterministic.
    #[test]
    fn test_nondeterministic_meas() {
        let mut state = SparseStabY::with_seed(1, 42);
        state.h(&q(0)); // |+>: stab = +X
        let r = state.mz(&q(0)).into_iter().next().unwrap();
        assert!(!r.is_deterministic);
    }

    /// Test Bell state measurement correlation: both qubits give same result.
    #[test]
    fn test_bell_state_measurement_correlation() {
        for seed in 0..100 {
            let mut state = SparseStabY::with_seed(2, seed);
            state.h(&q(0)).cx(&q2(0, 1));
            let r0 = state.mz(&q(0)).into_iter().next().unwrap();
            let r1 = state.mz(&q(1)).into_iter().next().unwrap();
            assert!(!r0.is_deterministic);
            assert!(r1.is_deterministic);
            assert_eq!(
                r0.outcome, r1.outcome,
                "Bell state qubits must agree (seed={seed})"
            );
        }
    }

    /// Convert a `CliffordRep` `PauliString` image to Y-convention representation.
    ///
    /// In Y-convention, both X and Z bits set = Y (the full Pauli, including the i).
    /// The `PauliString` phase is the coefficient of the named Pauli product, so
    /// no phase conversion is needed -- the stored phase equals `phase()` directly.
    ///
    /// Returns (`x_bits`, `z_bits`, `signs_minus`, `signs_i`) for `num_qubits` qubits.
    fn pauli_image_to_y_notation(
        image: &pecos_core::PauliString,
        num_qubits: usize,
    ) -> (Vec<bool>, Vec<bool>, bool, bool) {
        use pecos_core::Pauli;

        let mut x_bits = vec![false; num_qubits];
        let mut z_bits = vec![false; num_qubits];

        for (pauli, qid) in image.iter_pairs() {
            let qi = qid.index();
            match pauli {
                Pauli::I => {}
                Pauli::X => {
                    x_bits[qi] = true;
                }
                Pauli::Z => {
                    z_bits[qi] = true;
                }
                Pauli::Y => {
                    x_bits[qi] = true;
                    z_bits[qi] = true;
                }
            }
        }

        // In Y-convention, both bits set = Y directly (not W = XZ).
        // The CliffordRep phase is the coefficient of the Pauli product,
        // which maps directly to our stored phase with no conversion.
        let phase = image.phase() as u8;
        let signs_minus = phase & 1 != 0;
        let signs_i = phase & 2 != 0;
        (x_bits, z_bits, signs_minus, signs_i)
    }

    fn apply_1q_cliff(state: &mut SparseStabY, cliff: Clifford) {
        let qq = &[QubitId(0)];
        match cliff {
            Clifford::I => {}
            Clifford::X => {
                state.x(qq);
            }
            Clifford::Y => {
                state.y(qq);
            }
            Clifford::Z => {
                state.z(qq);
            }
            Clifford::SX => {
                state.sx(qq);
            }
            Clifford::SXdg => {
                state.sxdg(qq);
            }
            Clifford::SY => {
                state.sy(qq);
            }
            Clifford::SYdg => {
                state.sydg(qq);
            }
            Clifford::SZ => {
                state.sz(qq);
            }
            Clifford::SZdg => {
                state.szdg(qq);
            }
            Clifford::H => {
                state.h(qq);
            }
            Clifford::H2 => {
                state.h2(qq);
            }
            Clifford::H3 => {
                state.h3(qq);
            }
            Clifford::H4 => {
                state.h4(qq);
            }
            Clifford::H5 => {
                state.h5(qq);
            }
            Clifford::H6 => {
                state.h6(qq);
            }
            Clifford::F => {
                state.f(qq);
            }
            Clifford::Fdg => {
                state.fdg(qq);
            }
            Clifford::F2 => {
                state.f2(qq);
            }
            Clifford::F2dg => {
                state.f2dg(qq);
            }
            Clifford::F3 => {
                state.f3(qq);
            }
            Clifford::F3dg => {
                state.f3dg(qq);
            }
            Clifford::F4 => {
                state.f4(qq);
            }
            Clifford::F4dg => {
                state.f4dg(qq);
            }
            _ => panic!("Not a 1q gate: {cliff:?}"),
        }
    }

    fn apply_2q_cliff(state: &mut SparseStabY, cliff: Clifford) {
        let qq = &[(QubitId(0), QubitId(1))];
        match cliff {
            Clifford::CX => {
                state.cx(qq);
            }
            Clifford::CY => {
                state.cy(qq);
            }
            Clifford::CZ => {
                state.cz(qq);
            }
            Clifford::SXX => {
                state.sxx(qq);
            }
            Clifford::SXXdg => {
                state.sxxdg(qq);
            }
            Clifford::SYY => {
                state.syy(qq);
            }
            Clifford::SYYdg => {
                state.syydg(qq);
            }
            Clifford::SZZ => {
                state.szz(qq);
            }
            Clifford::SZZdg => {
                state.szzdg(qq);
            }
            Clifford::SWAP => {
                state.swap(qq);
            }
            Clifford::ISWAP => {
                state.iswap(qq);
            }
            Clifford::ISWAPdg => {
                state.iswapdg(qq);
            }
            Clifford::G => {
                state.g(qq);
            }
            Clifford::Gdg => {
                state.gdg(qq);
            }
            _ => panic!("Not a 2q gate: {cliff:?}"),
        }
    }

    /// Cross-check: `SparseStabY` gate images match `CliffordRep` for all 1q Cliffords.
    #[test]
    fn clifford_rep_matches_sparse_stab_y_all_1q_gates() {
        use pecos_core::PauliString;
        use pecos_core::clifford::Clifford;

        for &cliff in Clifford::all_1q() {
            let rep = cliff.on_qubit(0);

            for (name, input_ps, init_x) in [
                ("X", PauliString::x(0), true),
                ("Z", PauliString::z(0), false),
            ] {
                let image = rep.apply(&input_ps);
                let (exp_x, exp_z, exp_minus, exp_i) = pauli_image_to_y_notation(&image, 1);

                let mut state = SparseStabY::new(1);
                if init_x {
                    state.stabs.col_z[0].remove(0);
                    state.stabs.row_z[0].remove(0);
                    state.stabs.col_x[0].insert(0);
                    state.stabs.row_x[0].insert(0);
                }

                apply_1q_cliff(&mut state, cliff);

                assert_eq!(
                    state.stabs.col_x[0].contains(0),
                    exp_x[0],
                    "{cliff:?} on {name}: X bit mismatch (expected: {image:?})"
                );
                assert_eq!(
                    state.stabs.col_z[0].contains(0),
                    exp_z[0],
                    "{cliff:?} on {name}: Z bit mismatch (expected: {image:?})"
                );
                assert_eq!(
                    state.stabs.signs_minus.contains(0),
                    exp_minus,
                    "{cliff:?} on {name}: signs_minus mismatch (expected: {image:?})"
                );
                // In Y-convention, gates should NOT set signs_i
                assert!(
                    !state.stabs.signs_i.contains(0),
                    "{cliff:?} on {name}: signs_i should be empty for gates but was set. \
                     Y-convention expected signs_i={exp_i} (expected: {image:?})"
                );
            }
        }
    }

    /// Cross-check: `SparseStabY` gate images match `CliffordRep` for all 2q Cliffords.
    #[test]
    fn clifford_rep_matches_sparse_stab_y_all_2q_gates() {
        use pecos_core::PauliString;
        use pecos_core::clifford::Clifford;

        let inputs: [(&str, PauliString, usize, bool); 4] = [
            ("X0", PauliString::x(0), 0, true),
            ("Z0", PauliString::z(0), 0, false),
            ("X1", PauliString::x(1), 1, true),
            ("Z1", PauliString::z(1), 1, false),
        ];

        for &cliff in Clifford::all_2q() {
            let rep = cliff.on_qubits(0, 1);

            for (name, input_ps, input_q, init_x) in &inputs {
                let image = rep.apply(input_ps);
                let (exp_x, exp_z, exp_minus, exp_i) = pauli_image_to_y_notation(&image, 2);

                let mut state = SparseStabY::new(2);
                if *init_x {
                    state.stabs.col_z[*input_q].remove(*input_q);
                    state.stabs.row_z[*input_q].remove(*input_q);
                    state.stabs.col_x[*input_q].insert(*input_q);
                    state.stabs.row_x[*input_q].insert(*input_q);
                }

                apply_2q_cliff(&mut state, cliff);

                let gen_id = *input_q;
                for qubit in 0..2 {
                    assert_eq!(
                        state.stabs.col_x[qubit].contains(gen_id),
                        exp_x[qubit],
                        "{cliff:?} on {name}: qubit {qubit} X bit mismatch \
                         (expected image: {image:?})"
                    );
                    assert_eq!(
                        state.stabs.col_z[qubit].contains(gen_id),
                        exp_z[qubit],
                        "{cliff:?} on {name}: qubit {qubit} Z bit mismatch \
                         (expected image: {image:?})"
                    );
                }
                assert_eq!(
                    state.stabs.signs_minus.contains(gen_id),
                    exp_minus,
                    "{cliff:?} on {name}: signs_minus mismatch \
                     (expected image: {image:?})"
                );
                // In Y-convention, gates should NOT set signs_i
                assert!(
                    !state.stabs.signs_i.contains(gen_id),
                    "{cliff:?} on {name}: signs_i should be empty for gates but was set. \
                     Y-convention expected signs_i={exp_i} (expected: {image:?})"
                );
            }
        }
    }

    /// Test measurement after various gate sequences matches `SparseStab` (W-convention).
    #[test]
    fn test_measurement_matches_w_convention() {
        use crate::SparseStab;

        for seed in 0..50 {
            // Circuit 1: measure |0>
            {
                let mut y_sim = SparseStabY::with_seed(1, seed);
                let mut w_sim = SparseStab::with_seed(1, seed);
                let yr = y_sim.mz(&q(0));
                let wr = w_sim.mz(&q(0));
                assert_eq!(yr[0].outcome, wr[0].outcome, "meas |0> seed {seed}");
                assert_eq!(yr[0].is_deterministic, wr[0].is_deterministic);
            }

            // Circuit 2: X then measure (|1>)
            {
                let mut y_sim = SparseStabY::with_seed(1, seed);
                let mut w_sim = SparseStab::with_seed(1, seed);
                y_sim.x(&q(0));
                w_sim.x(&q(0));
                let yr = y_sim.mz(&q(0));
                let wr = w_sim.mz(&q(0));
                assert_eq!(yr[0].outcome, wr[0].outcome, "meas |1> seed {seed}");
                assert_eq!(yr[0].is_deterministic, wr[0].is_deterministic);
            }

            // Circuit 3: H then SZ then measure
            {
                let mut y_sim = SparseStabY::with_seed(1, seed);
                let mut w_sim = SparseStab::with_seed(1, seed);
                y_sim.h(&q(0)).sz(&q(0));
                w_sim.h(&q(0)).sz(&q(0));
                let yr = y_sim.mz(&q(0));
                let wr = w_sim.mz(&q(0));
                assert_eq!(yr[0].outcome, wr[0].outcome, "H+SZ seed {seed}");
                assert_eq!(yr[0].is_deterministic, wr[0].is_deterministic);
            }

            // Circuit 4: Bell state, measure both
            {
                let mut y_sim = SparseStabY::with_seed(2, seed);
                let mut w_sim = SparseStab::with_seed(2, seed);
                y_sim.h(&q(0)).cx(&q2(0, 1));
                w_sim.h(&q(0)).cx(&q2(0, 1));
                let yr = y_sim.mz(&[QubitId(0), QubitId(1)]);
                let wr = w_sim.mz(&[QubitId(0), QubitId(1)]);
                for i in 0..2 {
                    assert_eq!(yr[i].outcome, wr[i].outcome, "Bell q{i} seed {seed}");
                    assert_eq!(yr[i].is_deterministic, wr[i].is_deterministic);
                }
            }

            // Circuit 5: 3-qubit GHZ
            {
                let mut y_sim = SparseStabY::with_seed(3, seed);
                let mut w_sim = SparseStab::with_seed(3, seed);
                y_sim.h(&q(0)).cx(&q2(0, 1)).cx(&q2(1, 2));
                w_sim.h(&q(0)).cx(&q2(0, 1)).cx(&q2(1, 2));
                let yr = y_sim.mz(&[QubitId(0), QubitId(1), QubitId(2)]);
                let wr = w_sim.mz(&[QubitId(0), QubitId(1), QubitId(2)]);
                for i in 0..3 {
                    assert_eq!(yr[i].outcome, wr[i].outcome, "GHZ q{i} seed {seed}");
                    assert_eq!(yr[i].is_deterministic, wr[i].is_deterministic);
                }
            }
        }
    }

    /// Thorough cross-validation: run many random Clifford circuits on both
    /// `SparseStabY` and `SparseStab` and verify all measurements agree.
    /// Includes circuits with Y-stabilizers (SZ, SY, CY, SYY gates).
    #[test]
    fn thorough_measurement_cross_validation() {
        use crate::SparseStab;
        use pecos_core::clifford::Clifford;

        /// A test operation: either a single-qubit or two-qubit gate.
        enum Op {
            Sq(Clifford, QubitId),
            Tq(Clifford, QubitId, QubitId),
        }
        use Op::{Sq, Tq};

        fn apply(y: &mut SparseStabY, w: &mut SparseStab, op: &Op) {
            match *op {
                Sq(Clifford::H, q) => {
                    y.h(&[q]);
                    w.h(&[q]);
                }
                Sq(Clifford::SZ, q) => {
                    y.sz(&[q]);
                    w.sz(&[q]);
                }
                Sq(Clifford::SY, q) => {
                    y.sy(&[q]);
                    w.sy(&[q]);
                }
                Sq(g, _) => panic!("unhandled 1q gate {g:?}"),
                Tq(Clifford::CX, a, b) => {
                    y.cx(&[(a, b)]);
                    w.cx(&[(a, b)]);
                }
                Tq(Clifford::CY, a, b) => {
                    y.cy(&[(a, b)]);
                    w.cy(&[(a, b)]);
                }
                Tq(Clifford::CZ, a, b) => {
                    y.cz(&[(a, b)]);
                    w.cz(&[(a, b)]);
                }
                Tq(Clifford::SXX, a, b) => {
                    y.sxx(&[(a, b)]);
                    w.sxx(&[(a, b)]);
                }
                Tq(Clifford::SYY, a, b) => {
                    y.syy(&[(a, b)]);
                    w.syy(&[(a, b)]);
                }
                Tq(Clifford::SZZ, a, b) => {
                    y.szz(&[(a, b)]);
                    w.szz(&[(a, b)]);
                }
                Tq(Clifford::ISWAP, a, b) => {
                    y.iswap(&[(a, b)]);
                    w.iswap(&[(a, b)]);
                }
                Tq(g, _, _) => panic!("unhandled 2q gate {g:?}"),
            }
        }

        for seed in 0..100 {
            let mut y_sim = SparseStabY::with_seed(4, seed);
            let mut w_sim = SparseStab::with_seed(4, seed);

            // Build a circuit that creates Y-stabilizers
            // Pattern depends on seed to get variety
            let ops: Vec<Op> = match seed % 10 {
                0 => vec![
                    Sq(Clifford::H, QubitId(0)),
                    Sq(Clifford::SZ, QubitId(0)), // Creates Y stabilizer
                    Tq(Clifford::CX, QubitId(0), QubitId(1)),
                ],
                1 => vec![
                    Sq(Clifford::H, QubitId(0)),
                    Tq(Clifford::CY, QubitId(0), QubitId(1)), // Creates Y
                ],
                2 => vec![
                    Sq(Clifford::H, QubitId(0)),
                    Sq(Clifford::H, QubitId(1)),
                    Tq(Clifford::SYY, QubitId(0), QubitId(1)), // Y-Y entangling
                ],
                3 => vec![
                    Sq(Clifford::H, QubitId(0)),
                    Sq(Clifford::SZ, QubitId(0)),
                    Tq(Clifford::CX, QubitId(0), QubitId(1)),
                    Sq(Clifford::SZ, QubitId(1)),
                    Tq(Clifford::CX, QubitId(1), QubitId(2)),
                ],
                4 => vec![
                    Sq(Clifford::H, QubitId(0)),
                    Sq(Clifford::H, QubitId(2)),
                    Tq(Clifford::ISWAP, QubitId(0), QubitId(1)),
                    Tq(Clifford::CX, QubitId(2), QubitId(3)),
                ],
                5 => vec![
                    Sq(Clifford::H, QubitId(1)),
                    Sq(Clifford::SY, QubitId(1)),
                    Tq(Clifford::CZ, QubitId(0), QubitId(1)),
                    Sq(Clifford::H, QubitId(0)),
                ],
                6 => vec![
                    Sq(Clifford::H, QubitId(0)),
                    Tq(Clifford::CX, QubitId(0), QubitId(1)),
                    Sq(Clifford::SZ, QubitId(0)),
                    Sq(Clifford::SZ, QubitId(1)),
                    Sq(Clifford::H, QubitId(2)),
                    Tq(Clifford::CX, QubitId(2), QubitId(3)),
                ],
                7 => vec![
                    Sq(Clifford::H, QubitId(0)),
                    Sq(Clifford::SZ, QubitId(0)),
                    Sq(Clifford::H, QubitId(0)), // SZ H creates non-trivial 1q state
                    Tq(Clifford::CX, QubitId(0), QubitId(1)),
                ],
                8 => vec![
                    Sq(Clifford::H, QubitId(0)),
                    Sq(Clifford::H, QubitId(1)),
                    Tq(Clifford::CX, QubitId(0), QubitId(1)),
                    Sq(Clifford::SZ, QubitId(0)),
                    Sq(Clifford::SZ, QubitId(1)),
                    Tq(Clifford::CX, QubitId(1), QubitId(2)),
                    Sq(Clifford::SZ, QubitId(2)),
                ],
                _ => vec![
                    Sq(Clifford::H, QubitId(0)),
                    Tq(Clifford::SXX, QubitId(0), QubitId(1)),
                    Tq(Clifford::SYY, QubitId(1), QubitId(2)),
                    Tq(Clifford::SZZ, QubitId(2), QubitId(3)),
                ],
            };

            for op in &ops {
                apply(&mut y_sim, &mut w_sim, op);
            }

            let qubits_all = [QubitId(0), QubitId(1), QubitId(2), QubitId(3)];
            let yr = y_sim.mz(&qubits_all);
            let wr = w_sim.mz(&qubits_all);

            for i in 0..4 {
                assert_eq!(
                    yr[i].outcome,
                    wr[i].outcome,
                    "seed {seed} (circuit {}), qubit {i}: outcome mismatch",
                    seed % 10
                );
                assert_eq!(
                    yr[i].is_deterministic,
                    wr[i].is_deterministic,
                    "seed {seed} (circuit {}), qubit {i}: determinism mismatch",
                    seed % 10
                );
            }
        }
    }
}
