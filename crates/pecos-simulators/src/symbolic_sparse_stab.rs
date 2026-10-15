// Copyright 2025 The PECOS Developers
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

//! VecSet-based symbolic stabilizer simulator with measurement-indexed signs.
//!
//! This module provides [`SymbolicSparseStabVecSet`], a VecSet-based stabilizer simulator
//! that tracks measurement dependencies rather than collapsing to concrete outcomes.
//!
//! For large circuits (100+ qubits), consider using [`SymbolicSparseStab`](crate::SymbolicSparseStab)
//! which uses `BitSet` for faster performance.
//!
//! Instead of randomly choosing 0 or 1 for non-deterministic measurements, this simulator
//! assigns each measurement a unique index and tracks which measurements contribute to
//! each stabilizer's sign via XOR (symmetric difference).
//!
//! The measurement outcome is represented as: `{measurement_indices} ^ flip`
//! - `measurement_indices`: Set of measurement indices whose outcomes XOR together
//! - `flip`: Boolean indicating whether to flip the result (from unitary gate phases)

use crate::QuantumSimulator;
use crate::sign_algebra::{SignAlgebra, SymbolicSign};
use crate::symbolic_gens::SymbolicGensVecSet;
use core::mem;
use pecos_core::{BitSet, Set, VecSet};

/// Result of a symbolic measurement.
///
/// The outcome is represented as: `XOR(measurement_outcomes[i] for i in outcome) XOR flip`
///
/// For example:
/// - `outcome = {}, flip = false`: deterministic 0
/// - `outcome = {}, flip = true`: deterministic 1
/// - `outcome = {0}, flip = false`: same as measurement 0
/// - `outcome = {0}, flip = true`: opposite of measurement 0
/// - `outcome = {0, 2}, flip = false`: XOR of measurements 0 and 2
///
/// # Display Format
///
/// The `Display` implementation formats results as `m{index}={expression}`:
/// - `m0=?`: non-deterministic (random outcome)
/// - `m0=0`: deterministic 0 (no dependencies, no flip)
/// - `m0=1`: deterministic 1 (no dependencies, flip=true)
/// - `m2=m0`: measurement 2 equals measurement 0
/// - `m3=m2^m1`: measurement 3 equals m2 XOR m1
/// - `m3=m2^m1^1`: measurement 3 equals m2 XOR m1 XOR 1 (with flip)
///
/// Dependencies are ordered from largest to smallest index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicMeasurementResult {
    /// The set of measurement indices whose outcomes XOR together.
    /// Empty set means no measurement dependency (outcome is just `flip`).
    pub outcome: BitSet,
    /// Whether to flip the XOR result (accumulated from unitary gate phases).
    pub flip: bool,
    /// Whether this measurement was deterministic (outcome determined by prior measurements).
    pub is_deterministic: bool,
    /// The index assigned to this measurement.
    pub index: usize,
}

impl std::fmt::Display for SymbolicMeasurementResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Start with this measurement's index
        write!(f, "m{}=", self.index)?;

        if self.is_deterministic {
            if self.outcome.is_empty() {
                // No dependencies: just show flip as 0 or 1
                write!(f, "{}", u8::from(self.flip))
            } else {
                // Show dependencies in reverse order (largest to smallest)
                // Collect to Vec and reverse since BitSet iterates in ascending order
                let deps: Vec<_> = self.outcome.iter().collect();
                let mut first = true;
                for dep in deps.into_iter().rev() {
                    if !first {
                        write!(f, "^")?;
                    }
                    write!(f, "m{dep}")?;
                    first = false;
                }
                // Only add ^1 if flip is true
                if self.flip {
                    write!(f, "^1")?;
                }
                Ok(())
            }
        } else {
            // Non-deterministic: show as random/unknown
            write!(f, "?")
        }
    }
}

/// History of all measurements performed during simulation.
///
/// The index in the history equals the measurement index assigned to that measurement.
/// Provides methods to filter by deterministic/non-deterministic status.
#[derive(Clone, Debug, Default)]
pub struct MeasurementHistory {
    measurements: Vec<SymbolicMeasurementResult>,
}

impl MeasurementHistory {
    /// Create a new empty measurement history.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }

    /// Add a measurement result to the history.
    #[inline]
    pub fn push(&mut self, result: SymbolicMeasurementResult) {
        self.measurements.push(result);
    }

    /// Clear all measurements from the history.
    #[inline]
    pub fn clear(&mut self) {
        self.measurements.clear();
    }

    /// Returns the number of measurements in the history.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.measurements.len()
    }

    /// Returns true if the history is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }

    /// Returns a reference to the measurement at the given index.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&SymbolicMeasurementResult> {
        self.measurements.get(index)
    }

    /// Returns the full list of measurements as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[SymbolicMeasurementResult] {
        &self.measurements
    }

    /// Returns an iterator over all measurements.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &SymbolicMeasurementResult> {
        self.measurements.iter()
    }

    /// Returns only the deterministic measurements.
    ///
    /// Each result contains its measurement index in the `index` field.
    #[inline]
    #[must_use]
    pub fn deterministic(&self) -> Vec<&SymbolicMeasurementResult> {
        self.measurements
            .iter()
            .filter(|m| m.is_deterministic)
            .collect()
    }

    /// Returns only the non-deterministic measurements.
    ///
    /// Each result contains its measurement index in the `index` field.
    #[inline]
    #[must_use]
    pub fn nondeterministic(&self) -> Vec<&SymbolicMeasurementResult> {
        self.measurements
            .iter()
            .filter(|m| !m.is_deterministic)
            .collect()
    }

    /// Formats all measurements as a bracketed list.
    ///
    /// Example: `[m0^m0=0, m1^m0=0, m2=0]`
    #[must_use]
    pub fn format_all(&self) -> String {
        let formatted: Vec<String> = self.measurements.iter().map(ToString::to_string).collect();
        format!("[{}]", formatted.join(", "))
    }

    /// Formats only deterministic measurements as a bracketed list.
    ///
    /// Example: `[m1^m0=0, m2=0]`
    #[must_use]
    pub fn format_deterministic(&self) -> String {
        let formatted: Vec<String> = self
            .measurements
            .iter()
            .filter(|m| m.is_deterministic)
            .map(ToString::to_string)
            .collect();
        format!("[{}]", formatted.join(", "))
    }

    /// Formats only non-deterministic measurements as a bracketed list.
    ///
    /// Example: `[m0^m0=0, m3^m3=0]`
    #[must_use]
    pub fn format_nondeterministic(&self) -> String {
        let formatted: Vec<String> = self
            .measurements
            .iter()
            .filter(|m| !m.is_deterministic)
            .map(ToString::to_string)
            .collect();
        format!("[{}]", formatted.join(", "))
    }
}

impl std::fmt::Display for MeasurementHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_all())
    }
}

impl std::ops::Index<usize> for MeasurementHistory {
    type Output = SymbolicMeasurementResult;

    fn index(&self, index: usize) -> &Self::Output {
        &self.measurements[index]
    }
}

/// A symbolic stabilizer simulator that tracks measurement dependencies.
///
/// This simulator is based on the same stabilizer/destabilizer formalism as [`SparseStab`],
/// but instead of collapsing measurements to concrete outcomes, it tracks which measurements
/// contribute to each outcome.
///
/// # Type Parameters
/// - `T`: Set type for sparse storage with usize elements
///
/// # Use Cases
/// - Analyzing measurement dependency graphs
/// - Understanding which measurements affect which outcomes
/// - Pauli frame tracking / deferred measurement patterns
/// - Verifying measurement patterns in quantum error correction
///
/// # Example
/// ```rust
/// use pecos_simulators::symbolic_sparse_stab::SymbolicSparseStabVecSet;
/// use pecos_simulators::QuantumSimulator;
///
/// let mut sim = SymbolicSparseStabVecSet::new(2);
///
/// // Create Bell state
/// sim.h(&[0]).cx(&[(0, 1)]);
///
/// // Measure both qubits
/// let results = sim.mz(&[0]);  // Non-deterministic: outcome depends on measurement 0
/// let r0 = &results[0];
/// let results = sim.mz(&[1]);  // Deterministic: outcome equals measurement 0's outcome
/// let r1 = &results[0];
///
/// // r0.outcome = {0} (depends on measurement 0)
/// // r1.outcome = {0} (also depends on measurement 0, showing correlation)
/// assert!(!r0.is_deterministic);
/// assert!(r1.is_deterministic);
/// assert_eq!(r0.outcome, r1.outcome);  // Same dependency = correlated
/// ```
#[derive(Clone, Debug)]
pub struct SymbolicSparseStabVecSet {
    num_qubits: usize,
    stabs: SymbolicGensVecSet,
    destabs: SymbolicGensVecSet,
    /// Counter for assigning unique indices to measurements
    measurement_counter: usize,
    /// History of all measurements performed.
    measurement_history: MeasurementHistory,
}

impl SymbolicSparseStabVecSet {
    /// Create a new symbolic stabilizer simulator.
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let mut sim = Self {
            num_qubits,
            stabs: SymbolicGensVecSet::new(num_qubits),
            destabs: SymbolicGensVecSet::new(num_qubits),
            measurement_counter: 0,
            measurement_history: MeasurementHistory::new(),
        };
        sim.reset();
        sim
    }

    /// Returns the number of qubits in the system.
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the current measurement counter (number of measurements made so far).
    #[inline]
    #[must_use]
    pub fn measurement_count(&self) -> usize {
        self.measurement_counter
    }

    /// Returns a reference to the measurement history.
    ///
    /// The history provides methods to access all measurements, or filter by
    /// deterministic/non-deterministic status.
    ///
    /// # Example
    /// ```rust
    /// use pecos_simulators::symbolic_sparse_stab::SymbolicSparseStabVecSet;
    ///
    /// let mut sim = SymbolicSparseStabVecSet::new(2);
    /// sim.h(&[0]).cx(&[(0, 1)]);
    /// sim.mz(&[0]);
    /// sim.mz(&[1]);
    ///
    /// let history = sim.measurement_history();
    /// assert_eq!(history.len(), 2);
    ///
    /// // Get deterministic measurements (each result has its index in result.index)
    /// let det = history.deterministic();
    /// for result in det {
    ///     println!("Measurement {}: {:?}", result.index, result.outcome);
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn measurement_history(&self) -> &MeasurementHistory {
        &self.measurement_history
    }

    /// Produces a textual representation of the stabilizer tableau with symbolic signs.
    ///
    /// Format: `{measurement_indices} PauliString`
    /// Example: `{} ZII` means identity sign (deterministic 0), `{0,1} XIZ` means XOR of measurements 0 and 1.
    #[must_use]
    pub fn stab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.stabs)
    }

    /// Produces a textual representation of the destabilizer tableau with symbolic signs.
    #[must_use]
    pub fn destab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.destabs)
    }

    /// Utility that creates a string representation of generators with symbolic signs.
    ///
    /// Format: `{measurement_indices} ^ flip PauliString`
    /// - `{measurement_indices}`: Set of measurement indices whose outcomes XOR together
    /// - `flip`: 0 or 1 indicating whether to flip the result (from unitary phases)
    ///
    /// Examples:
    /// - `{} ^ 0 ZII`: Identity sign, no flip (deterministic 0)
    /// - `{} ^ 1 ZII`: Identity sign, flipped (deterministic 1)
    /// - `{0} ^ 0 XIZ`: Depends on measurement 0, no flip
    /// - `{0,1} ^ 1 XIZ`: XOR of measurements 0,1, flipped
    fn tableau_string(num_qubits: usize, gens: &SymbolicGensVecSet) -> String {
        use std::fmt::Write;

        let mut result = String::new();
        for i in 0..num_qubits {
            // Compute the flip from signs_minus and signs_i
            // For stabilizers, we only care about the minus component
            // (imaginary components should cancel out in valid stabilizer states)
            let has_minus = gens.signs_minus.contains(&i);
            let flip = u8::from(has_minus);

            // Format the symbolic sign with flip
            let sign = &gens.signs[i];
            if sign.measurements.is_empty() {
                let _ = write!(result, "{{}} ^ {flip} ");
            } else {
                let indices: Vec<String> = sign
                    .measurements
                    .iter()
                    .map(|idx| idx.to_string())
                    .collect();
                let _ = write!(result, "{{{}}} ^ {flip} ", indices.join(","));
            }

            // Format the Pauli string
            for qubit in 0..num_qubits {
                let in_row_x = gens.row_x[i].contains(&qubit);
                let in_row_z = gens.row_z[i].contains(&qubit);

                let c = match (in_row_x, in_row_z) {
                    (false, false) => 'I',
                    (true, false) => 'X',
                    (false, true) => 'Z',
                    (true, true) => 'Y',
                };
                result.push(c);
            }
            result.push('\n');
        }
        result
    }

    /// Reset the simulator to the initial |00...0⟩ state.
    #[inline]
    pub fn reset(&mut self) -> &mut Self {
        self.stabs.init_all_z();
        self.destabs.init_all_x();
        self.measurement_counter = 0;
        self.measurement_history.clear();
        self
    }

    // ==================== Gate Operations ====================
    // Same as SparseStab, tracking phase changes via signs_minus/signs_i

    /// Pauli X gate. X -> X, Z -> -Z
    #[inline]
    pub fn x(&mut self, qubits: &[usize]) -> &mut Self {
        for &q in qubits {
            self.stabs.signs_minus ^= &self.stabs.col_z[q];
        }
        self
    }

    /// Pauli Y gate. X -> -X, Z -> -Z
    #[inline]
    pub fn y(&mut self, qubits: &[usize]) -> &mut Self {
        for &q in qubits {
            for i in self.stabs.col_x[q].symmetric_difference(&self.stabs.col_z[q]) {
                self.stabs.signs_minus ^= i;
            }
        }
        self
    }

    /// Pauli Z gate. X -> -X, Z -> Z
    #[inline]
    pub fn z(&mut self, qubits: &[usize]) -> &mut Self {
        for &q in qubits {
            self.stabs.signs_minus ^= &self.stabs.col_x[q];
        }
        self
    }

    /// Sqrt of Z gate (S gate). X -> Y, Z -> Z
    #[inline]
    pub fn sz(&mut self, qubits: &[usize]) -> &mut Self {
        for &q in qubits {
            // X -> i: track phase changes
            // i * i = -1, so if already has i, add minus and remove i
            for i in self.stabs.signs_i.intersection(&self.stabs.col_x[q]) {
                self.stabs.signs_minus ^= i;
            }
            self.stabs.signs_i ^= &self.stabs.col_x[q];

            // Update the Pauli structure (X -> Y means add Z component)
            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_z[q] ^= &g.col_x[q];

                for &i in &g.col_x[q] {
                    g.row_z[i] ^= &q;
                }
            }
        }
        self
    }

    /// Hadamard gate. X -> Z, Z -> X, Y -> -Y
    #[inline]
    pub fn h(&mut self, qubits: &[usize]) -> &mut Self {
        for &q in qubits {
            // Y -> -Y: add minus for generators that have both X and Z on this qubit
            for i in self.stabs.col_x[q].intersection(&self.stabs.col_z[q]) {
                self.stabs.signs_minus ^= i;
            }

            // Swap X and Z for this qubit
            for g in [&mut self.stabs, &mut self.destabs] {
                for i in g.col_x[q].difference(&g.col_z[q]) {
                    g.row_x[*i].remove(&q);
                    g.row_z[*i].insert(q);
                }

                for i in g.col_z[q].difference(&g.col_x[q]) {
                    g.row_z[*i].remove(&q);
                    g.row_x[*i].insert(q);
                }

                mem::swap(&mut g.col_x[q], &mut g.col_z[q]);
            }
        }
        self
    }

    /// CNOT gate. IX -> IX, XI -> XX, IZ -> ZZ, ZI -> ZI
    #[inline]
    pub fn cx(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        for &(q1, q2) in pairs {
            for g in &mut [&mut self.stabs, &mut self.destabs] {
                let (qu_min, qu_max) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

                // Handle col_x: XI -> XX
                {
                    let (_left, right) = g.col_x.split_at_mut(qu_min);
                    let (mid, right) = right.split_at_mut(qu_max - qu_min);
                    let col_x_min = &mut mid[0];
                    let col_x_max = &mut right[0];

                    let (col_x_qu1, col_x_qu2) = if q1 < q2 {
                        (col_x_min, col_x_max)
                    } else {
                        (col_x_max, col_x_min)
                    };

                    let mut q2_set = VecSet::new();
                    q2_set.insert(q2);

                    for i in col_x_qu1.iter() {
                        g.row_x[*i].symmetric_difference_update(&q2_set);
                    }
                    col_x_qu2.symmetric_difference_update(col_x_qu1);
                }

                // Handle col_z: IZ -> ZZ
                {
                    let (_left, right) = g.col_z.split_at_mut(qu_min);
                    let (mid, right) = right.split_at_mut(qu_max - qu_min);
                    let col_z_min = &mut mid[0];
                    let col_z_max = &mut right[0];

                    let (col_z_qu1, col_z_qu2) = if q1 < q2 {
                        (col_z_min, col_z_max)
                    } else {
                        (col_z_max, col_z_min)
                    };

                    let mut q1_set = VecSet::new();
                    q1_set.insert(q1);

                    for i in col_z_qu2.iter() {
                        g.row_z[*i].symmetric_difference_update(&q1_set);
                    }
                    col_z_qu1.symmetric_difference_update(col_z_qu2);
                }
            }
        }
        self
    }

    // ==================== Measurement ====================

    /// Measure qubits in the Z basis.
    ///
    /// Returns a `Vec<SymbolicMeasurementResult>` containing the set of measurement indices
    /// whose outcomes XOR together to determine each measurement's result.
    #[inline]
    pub fn mz(&mut self, qubits: &[usize]) -> Vec<SymbolicMeasurementResult> {
        qubits
            .iter()
            .map(|&q| {
                let result = if self.stabs.col_x[q].is_empty() {
                    // Deterministic measurement
                    self.deterministic_meas(q)
                } else {
                    // Non-deterministic measurement
                    self.nondeterministic_meas(q)
                };

                // Record in measurement history
                self.measurement_history.push(result.clone());

                result
            })
            .collect()
    }

    /// Prepare qubit in |0⟩ (reset). Does not record a measurement.
    ///
    /// Physically: measure Z, discard the outcome, conditionally apply X
    /// to force the +1 eigenvalue.
    ///
    /// Symbolically: project qubit onto +Z eigenstate with empty sign
    /// (no measurement dependencies). If the qubit is already in a Z
    /// eigenstate, H rotates it to the X basis first so the non-deterministic
    /// projection path can properly disentangle it from all other qubits.
    pub fn pz(&mut self, q: usize) -> &mut Self {
        if self.stabs.col_x[q].is_empty() {
            // Qubit is in a Z eigenstate. Rotate to X basis so the
            // non-deterministic projection correctly disentangles it
            // and transfers sign information through the stabilizer group.
            self.h(&[q]);
        }
        self.pz_nondeterministic(q);
        self
    }

    /// Project qubit onto +Z eigenstate without recording a measurement.
    ///
    /// Same Gaussian elimination as `nondeterministic_meas` but does not
    /// record a measurement or increment the counter. The resulting `Z` on `q`
    /// stabilizer gets an empty sign (eigenvalue +1).
    fn pz_nondeterministic(&mut self, q: usize) {
        let mut anticom_stabs_col = self.stabs.col_x[q].clone();
        let mut anticom_destabs_col = self.destabs.col_x[q].clone();

        // Find stabilizer to replace (smallest weight)
        let mut smallest_wt = 2 * self.num_qubits + 2;
        let mut removed_id: Option<usize> = None;

        for stab_id in &anticom_stabs_col {
            let weight = self.stabs.row_x[*stab_id].len() + self.stabs.row_z[*stab_id].len();
            if weight < smallest_wt {
                smallest_wt = weight;
                removed_id = Some(*stab_id);
            }
        }

        let id = removed_id.expect("col_x[q] was non-empty");
        anticom_stabs_col.remove(&id);
        let removed_row_x = self.stabs.row_x[id].clone();
        let removed_row_z = self.stabs.row_z[id].clone();

        // Phase tracking for anticommuting stabilizers
        if self.stabs.signs_minus.contains(&id) {
            self.stabs.signs_minus ^= &anticom_stabs_col;
        }
        if self.stabs.signs_i.contains(&id) {
            self.stabs.signs_i.remove(&id);
            let gens_common: Vec<_> = self
                .stabs
                .signs_i
                .intersection(&anticom_stabs_col)
                .copied()
                .collect();
            let gens_only_stabs: Vec<_> = anticom_stabs_col
                .difference(&self.stabs.signs_i)
                .copied()
                .collect();
            for i in gens_common {
                self.stabs.signs_minus ^= &i;
                self.stabs.signs_i.remove(&i);
            }
            for i in gens_only_stabs {
                self.stabs.signs_i.insert(i);
            }
        }

        // Multiply all other anticommuting stabilizers by the removed one
        let removed_sign = self.stabs.signs[id].clone();
        for g in &anticom_stabs_col {
            self.stabs.signs[*g].multiply_assign(&removed_sign);
            let num_minuses = removed_row_z.intersection(&self.stabs.row_x[*g]).count();
            if num_minuses & 1 != 0 {
                self.stabs.signs_minus ^= g;
            }
            self.stabs.row_x[*g] ^= &removed_row_x;
            self.stabs.row_z[*g] ^= &removed_row_z;
        }

        // Update column storage for stabilizers
        for i in &removed_row_x {
            self.stabs.col_x[*i] ^= &anticom_stabs_col;
        }
        for i in &removed_row_z {
            self.stabs.col_z[*i] ^= &anticom_stabs_col;
        }

        // Remove old stabilizer
        for i in &self.stabs.row_x[id] {
            self.stabs.col_x[*i].remove(&id);
        }
        for i in &self.stabs.row_z[id] {
            self.stabs.col_z[*i].remove(&id);
        }

        // Replace with Z_q, sign = empty (forced +1 eigenvalue)
        self.stabs.col_z[q].insert(id);
        self.stabs.row_x[id].clear();
        self.stabs.row_z[id].clear();
        self.stabs.row_z[id].insert(q);
        self.stabs.signs[id] = SymbolicSign::empty();
        self.stabs.signs_minus.remove(&id);
        self.stabs.signs_i.remove(&id);

        // Update destabilizers
        for i in &self.destabs.row_x[id] {
            self.destabs.col_x[*i].remove(&id);
        }
        for i in &self.destabs.row_z[id] {
            self.destabs.col_z[*i].remove(&id);
        }

        anticom_destabs_col.remove(&id);
        for i in &removed_row_x {
            self.destabs.col_x[*i].insert(id);
            self.destabs.col_x[*i] ^= &anticom_destabs_col;
        }
        for i in &removed_row_z {
            self.destabs.col_z[*i].insert(id);
            self.destabs.col_z[*i] ^= &anticom_destabs_col;
        }
        for row in &anticom_destabs_col {
            self.destabs.row_x[*row] ^= &removed_row_x;
            self.destabs.row_z[*row] ^= &removed_row_z;
        }
        self.destabs.row_x[id] = removed_row_x;
        self.destabs.row_z[id] = removed_row_z;
    }

    /// Handle a deterministic measurement.
    /// The outcome is determined by combining:
    /// 1. XOR of measurement dependencies from contributing stabilizers
    /// 2. Phase flip from `signs_minus` and `signs_i` (same logic as `SparseStab`)
    fn deterministic_meas(&mut self, q: usize) -> SymbolicMeasurementResult {
        // Assign index and increment counter
        let index = self.measurement_counter;
        self.measurement_counter += 1;

        // --- Phase flip calculation (from SparseStab) ---
        // Count minuses from destabilizers that anti-commute with Z_q
        let mut num_minuses = self.destabs.col_x[q]
            .intersection(&self.stabs.signs_minus)
            .count();

        let num_is = self.destabs.col_x[q]
            .intersection(&self.stabs.signs_i)
            .count();

        // Account for Pauli multiplication phases
        let mut cumulative_x = VecSet::new();
        for row in &self.destabs.col_x[q] {
            num_minuses += self.stabs.row_z[*row].intersection(&cumulative_x).count();
            cumulative_x ^= &self.stabs.row_x[*row];
        }

        if num_is & 3 != 0 {
            // num_is % 4 != 0
            num_minuses += 1;
        }

        let flip = num_minuses & 1 != 0; // num_minuses % 2 != 0 (is odd)

        // --- Measurement dependencies ---
        // XOR together the symbolic signs of all stabilizers corresponding to destabilizers
        // that have X on qubit q
        let mut result_sign = SymbolicSign::empty();

        for row in &self.destabs.col_x[q] {
            result_sign.multiply_assign(&self.stabs.signs[*row]);
        }

        SymbolicMeasurementResult {
            outcome: result_sign.measurements,
            flip,
            is_deterministic: true,
            index,
        }
    }

    /// Handle a non-deterministic measurement.
    /// Assigns a new measurement index and updates the stabilizer tableau.
    #[allow(clippy::too_many_lines)]
    fn nondeterministic_meas(&mut self, q: usize) -> SymbolicMeasurementResult {
        // Non-deterministic measurements always get an index (required for tracking)
        let measurement_index = self.measurement_counter;
        self.measurement_counter += 1;

        let mut anticom_stabs_col = self.stabs.col_x[q].clone();
        let mut anticom_destabs_col = self.destabs.col_x[q].clone();

        // Find a stabilizer to replace (choose smallest weight for efficiency)
        let mut smallest_wt = 2 * self.num_qubits + 2;
        let mut removed_id: Option<usize> = None;

        for stab_id in &anticom_stabs_col {
            let weight = self.stabs.row_x[*stab_id].len() + self.stabs.row_z[*stab_id].len();

            if weight < smallest_wt {
                smallest_wt = weight;
                removed_id = Some(*stab_id);
            }
        }

        let id = removed_id.expect("Critical error: removed_id was None");
        anticom_stabs_col.remove(&id);
        let removed_row_x = self.stabs.row_x[id].clone();
        let removed_row_z = self.stabs.row_z[id].clone();

        // --- Phase tracking for anticommuting stabilizers (from SparseStab) ---
        // If removed stabilizer has minus sign, propagate to others
        if self.stabs.signs_minus.contains(&id) {
            self.stabs.signs_minus ^= &anticom_stabs_col;
        }

        // Handle imaginary component propagation
        if self.stabs.signs_i.contains(&id) {
            self.stabs.signs_i.remove(&id);

            let gens_common: Vec<_> = self
                .stabs
                .signs_i
                .intersection(&anticom_stabs_col)
                .copied()
                .collect();
            let gens_only_stabs: Vec<_> = anticom_stabs_col
                .difference(&self.stabs.signs_i)
                .copied()
                .collect();

            for i in gens_common {
                self.stabs.signs_minus ^= &i;
                self.stabs.signs_i.remove(&i);
            }

            for i in gens_only_stabs {
                self.stabs.signs_i.insert(i);
            }
        }

        // Multiply all other anticommuting stabilizers by the removed one
        // This includes both measurement dependencies AND phase from Pauli multiplication
        let removed_sign = self.stabs.signs[id].clone();
        for g in &anticom_stabs_col {
            // Multiply the symbolic measurement signs (XOR)
            self.stabs.signs[*g].multiply_assign(&removed_sign);

            // Track phase from Pauli multiplication
            let num_minuses = removed_row_z.intersection(&self.stabs.row_x[*g]).count();
            if num_minuses & 1 != 0 {
                self.stabs.signs_minus ^= g;
            }

            // Update the Pauli structure
            self.stabs.row_x[*g] ^= &removed_row_x;
            self.stabs.row_z[*g] ^= &removed_row_z;
        }

        // Update column storage for stabilizers
        for i in &removed_row_x {
            self.stabs.col_x[*i] ^= &anticom_stabs_col;
        }

        for i in &removed_row_z {
            self.stabs.col_z[*i] ^= &anticom_stabs_col;
        }

        // Remove the old stabilizer
        for i in &self.stabs.row_x[id] {
            self.stabs.col_x[*i].remove(&id);
        }

        for i in &self.stabs.row_z[id] {
            self.stabs.col_z[*i].remove(&id);
        }

        // Replace with the measured stabilizer Z_q
        self.stabs.col_z[q].insert(id);
        self.stabs.row_x[id].clear();
        self.stabs.row_z[id].clear();
        self.stabs.row_z[id].insert(q);

        // Set the sign of the new stabilizer to this measurement's index
        // Also clear any phase tracking for this stabilizer (fresh start)
        self.stabs.signs[id] = SymbolicSign::single(measurement_index);
        self.stabs.signs_minus.remove(&id);
        self.stabs.signs_i.remove(&id);

        // Update destabilizers
        for i in &self.destabs.row_x[id] {
            self.destabs.col_x[*i].remove(&id);
        }

        for i in &self.destabs.row_z[id] {
            self.destabs.col_z[*i].remove(&id);
        }

        anticom_destabs_col.remove(&id);

        for i in &removed_row_x {
            self.destabs.col_x[*i].insert(id);
            self.destabs.col_x[*i] ^= &anticom_destabs_col;
        }

        for i in &removed_row_z {
            self.destabs.col_z[*i].insert(id);
            self.destabs.col_z[*i] ^= &anticom_destabs_col;
        }

        for row in &anticom_destabs_col {
            self.destabs.row_x[*row] ^= &removed_row_x;
            self.destabs.row_z[*row] ^= &removed_row_z;
        }

        self.destabs.row_x[id] = removed_row_x;
        self.destabs.row_z[id] = removed_row_z;

        // The outcome is just this measurement's index, with no flip
        // (the measurement result is "fresh" - no accumulated phase)
        SymbolicMeasurementResult {
            outcome: BitSet::single(measurement_index),
            flip: false,
            is_deterministic: false,
            index: measurement_index,
        }
    }
}

impl QuantumSimulator for SymbolicSparseStabVecSet {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    #[inline]
    fn reset(&mut self) -> &mut Self {
        Self::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bell_state_symbolic() {
        let mut sim = SymbolicSparseStabVecSet::new(2);

        // Create Bell state
        sim.h(&[0]).cx(&[(0, 1)]);

        // Measure qubit 0 - should be non-deterministic
        let r0 = sim.mz(&[0])[0].clone();
        assert!(!r0.is_deterministic);
        assert_eq!(r0.outcome.len(), 1);
        assert!(r0.outcome.contains(0)); // First measurement has index 0
        assert_eq!(r0.index, 0);

        // Measure qubit 1 - should be deterministic but still gets an index
        let r1 = sim.mz(&[1])[0].clone();
        assert!(r1.is_deterministic);
        assert_eq!(r0.outcome, r1.outcome); // Same measurement dependency = correlated
        assert_eq!(r1.index, 1);
    }

    #[test]
    fn test_product_state_symbolic() {
        let mut sim = SymbolicSparseStabVecSet::new(2);

        // Measure qubit 0 without any gates - should be deterministic |0⟩
        let r0 = sim.mz(&[0])[0].clone();
        assert!(r0.is_deterministic);
        assert!(r0.outcome.is_empty()); // Empty set = deterministic 0
        assert_eq!(r0.index, 0);

        // Measure qubit 1 - also deterministic |0⟩
        let r1 = sim.mz(&[1])[0].clone();
        assert!(r1.is_deterministic);
        assert!(r1.outcome.is_empty());
        assert_eq!(r1.index, 1);
    }

    #[test]
    fn test_hadamard_measurement_symbolic() {
        let mut sim = SymbolicSparseStabVecSet::new(1);

        // Apply H to put in superposition
        sim.h(&[0]);

        // Measure - should be non-deterministic
        let r = sim.mz(&[0])[0].clone();
        assert!(!r.is_deterministic);
        assert_eq!(r.outcome.len(), 1);
        assert!(r.outcome.contains(0));
        assert_eq!(r.index, 0);
    }

    #[test]
    fn test_ghz_state_symbolic() {
        let mut sim = SymbolicSparseStabVecSet::new(3);

        // Create GHZ state: (|000⟩ + |111⟩)/√2
        sim.h(&[0]).cx(&[(0, 1)]).cx(&[(1, 2)]);

        // Measure qubit 0 - non-deterministic
        let r0 = sim.mz(&[0])[0].clone();
        assert!(!r0.is_deterministic);
        assert!(r0.outcome.contains(0));
        assert_eq!(r0.index, 0);

        // Measure qubit 1 - deterministic, depends on measurement 0
        let r1 = sim.mz(&[1])[0].clone();
        assert!(r1.is_deterministic);
        assert_eq!(r0.outcome, r1.outcome);
        assert_eq!(r1.index, 1);

        // Measure qubit 2 - deterministic, depends on measurement 0
        let r2 = sim.mz(&[2])[0].clone();
        assert!(r2.is_deterministic);
        assert_eq!(r0.outcome, r2.outcome);
        assert_eq!(r2.index, 2);
    }

    #[test]
    fn test_multiple_independent_measurements() {
        let mut sim = SymbolicSparseStabVecSet::new(2);

        // Put both qubits in superposition independently
        sim.h(&[0]).h(&[1]);

        // Measure qubit 0 - non-deterministic, index 0
        let r0 = sim.mz(&[0])[0].clone();
        assert!(!r0.is_deterministic);
        assert!(r0.outcome.contains(0));
        assert_eq!(r0.index, 0);

        // Measure qubit 1 - non-deterministic, index 1
        let r1 = sim.mz(&[1])[0].clone();
        assert!(!r1.is_deterministic);
        assert!(r1.outcome.contains(1));
        assert_eq!(r1.index, 1);

        // They should have different measurement indices (independent)
        assert_ne!(r0.outcome, r1.outcome);
    }

    #[test]
    fn test_measurement_counter() {
        let mut sim = SymbolicSparseStabVecSet::new(3);
        assert_eq!(sim.measurement_count(), 0);

        // All deterministic measurements - counter always increments
        sim.mz(&[0]);
        assert_eq!(sim.measurement_count(), 1);

        sim.mz(&[1]);
        assert_eq!(sim.measurement_count(), 2);

        sim.mz(&[2]);
        assert_eq!(sim.measurement_count(), 3);
    }

    #[test]
    fn test_measurement_counter_with_nondet() {
        let mut sim = SymbolicSparseStabVecSet::new(3);
        assert_eq!(sim.measurement_count(), 0);

        // Make non-deterministic measurements
        sim.h(&[0]).h(&[1]).h(&[2]);

        sim.mz(&[0]);
        assert_eq!(sim.measurement_count(), 1);

        sim.mz(&[1]);
        assert_eq!(sim.measurement_count(), 2);

        sim.mz(&[2]);
        assert_eq!(sim.measurement_count(), 3);
    }

    #[test]
    fn test_deterministic_flag() {
        let mut sim = SymbolicSparseStabVecSet::new(2);

        // Deterministic measurement on |0⟩
        let r0 = sim.mz(&[0])[0].clone();
        assert!(r0.is_deterministic);

        // Reset and make non-deterministic
        sim.reset();
        sim.h(&[0]);
        let r1 = sim.mz(&[0])[0].clone();
        assert!(!r1.is_deterministic);
    }

    #[test]
    fn test_x_gate_flip() {
        // User's example: start with |0⟩, apply X to flip to |1⟩
        // The stabilizer goes from +Z to -Z, so outcome should be {} ^ 1
        let mut sim = SymbolicSparseStabVecSet::new(1);

        // Initial state: stabilized by +Z
        assert_eq!(sim.stab_tableau(), "{} ^ 0 Z\n");

        // Apply X: stabilizer becomes -Z
        sim.x(&[0]);
        assert_eq!(sim.stab_tableau(), "{} ^ 1 Z\n");

        // Measure - should be deterministic 1 (empty set XOR 1 = 1)
        let r = sim.mz(&[0])[0].clone();
        assert!(r.is_deterministic);
        assert!(r.outcome.is_empty()); // No measurement dependencies
        assert!(r.flip); // Flip is true, so result is 1
    }

    #[test]
    fn test_y_gate_flip() {
        // Y gate: X -> -X, Z -> -Z
        let mut sim = SymbolicSparseStabVecSet::new(1);

        // Apply Y: +Z becomes -Z
        sim.y(&[0]);
        assert_eq!(sim.stab_tableau(), "{} ^ 1 Z\n");

        let r = sim.mz(&[0])[0].clone();
        assert!(r.is_deterministic);
        assert!(r.outcome.is_empty());
        assert!(r.flip);
    }

    #[test]
    fn test_z_gate_no_flip() {
        // Z gate: X -> -X, Z -> Z (no change to Z stabilizer)
        let mut sim = SymbolicSparseStabVecSet::new(1);

        // Initial state: +Z
        assert_eq!(sim.stab_tableau(), "{} ^ 0 Z\n");

        // Apply Z: +Z stays +Z
        sim.z(&[0]);
        assert_eq!(sim.stab_tableau(), "{} ^ 0 Z\n");

        let r = sim.mz(&[0])[0].clone();
        assert!(r.is_deterministic);
        assert!(r.outcome.is_empty());
        assert!(!r.flip); // No flip
    }

    #[test]
    fn test_double_x_cancels() {
        // X X = I, so two X gates should cancel
        let mut sim = SymbolicSparseStabVecSet::new(1);

        sim.x(&[0]);
        assert_eq!(sim.stab_tableau(), "{} ^ 1 Z\n");

        sim.x(&[0]);
        assert_eq!(sim.stab_tableau(), "{} ^ 0 Z\n");

        let r = sim.mz(&[0])[0].clone();
        assert!(r.is_deterministic);
        assert!(r.outcome.is_empty());
        assert!(!r.flip);
    }

    #[test]
    fn test_hadamard_then_measure() {
        // H on |0⟩ gives |+⟩ which is non-deterministic
        let mut sim = SymbolicSparseStabVecSet::new(1);

        // H transforms Z stabilizer to X stabilizer
        sim.h(&[0]);
        assert_eq!(sim.stab_tableau(), "{} ^ 0 X\n");

        // Measure - non-deterministic, no flip since no accumulated phase
        let r = sim.mz(&[0])[0].clone();
        assert!(!r.is_deterministic);
        assert!(r.outcome.contains(0));
        assert!(!r.flip);
    }

    #[test]
    fn test_flip_propagates_through_bell() {
        // Create Bell state with an X gate first to introduce a flip
        let mut sim = SymbolicSparseStabVecSet::new(2);

        // Apply X to qubit 0 first, then create Bell state
        sim.x(&[0]).h(&[0]).cx(&[(0, 1)]);

        // Measure qubit 0
        let r0 = sim.mz(&[0])[0].clone();
        assert!(!r0.is_deterministic);
        assert!(r0.outcome.contains(0));
        // The X gate introduces a flip that should propagate
        // Note: The exact flip value depends on how phases propagate through H and CX

        // Measure qubit 1 - should be correlated with qubit 0
        let r1 = sim.mz(&[1])[0].clone();
        assert!(r1.is_deterministic);
        assert_eq!(r0.outcome, r1.outcome);
    }

    #[test]
    fn test_tableau_format() {
        let mut sim = SymbolicSparseStabVecSet::new(2);

        // Initial state
        let tableau = sim.stab_tableau();
        assert!(tableau.contains("{} ^ 0"));
        assert!(tableau.contains("ZI"));
        assert!(tableau.contains("IZ"));

        // After X on qubit 0
        sim.x(&[0]);
        let tableau = sim.stab_tableau();
        assert!(tableau.contains("{} ^ 1 ZI")); // First stabilizer gets flipped
        assert!(tableau.contains("{} ^ 0 IZ")); // Second stabilizer unchanged
    }

    #[test]
    fn test_measurement_history() {
        let mut sim = SymbolicSparseStabVecSet::new(3);

        // Initially no measurements
        assert!(sim.measurement_history().is_empty());

        // Create GHZ state and measure
        sim.h(&[0]).cx(&[(0, 1)]).cx(&[(1, 2)]);

        let r0 = sim.mz(&[0])[0].clone();
        assert_eq!(sim.measurement_history().len(), 1);
        assert_eq!(sim.measurement_history()[0], r0);

        let r1 = sim.mz(&[1])[0].clone();
        assert_eq!(sim.measurement_history().len(), 2);
        assert_eq!(sim.measurement_history()[1], r1);

        let r2 = sim.mz(&[2])[0].clone();
        assert_eq!(sim.measurement_history().len(), 3);
        assert_eq!(sim.measurement_history()[2], r2);

        // Check indices match positions
        assert_eq!(sim.measurement_history()[0].index, 0);
        assert_eq!(sim.measurement_history()[1].index, 1);
        assert_eq!(sim.measurement_history()[2].index, 2);

        // Check deterministic filtering
        let det = sim.measurement_history().deterministic();
        let nondet = sim.measurement_history().nondeterministic();

        // First measurement is non-deterministic, others are deterministic
        assert_eq!(nondet.len(), 1);
        assert_eq!(det.len(), 2);
        // The result contains its own index
        assert_eq!(nondet[0].index, 0);
        assert!(!nondet[0].is_deterministic);
    }

    #[test]
    fn test_measurement_history_reset() {
        let mut sim = SymbolicSparseStabVecSet::new(2);

        sim.h(&[0]);
        sim.mz(&[0]);
        sim.mz(&[1]);
        assert_eq!(sim.measurement_history().len(), 2);

        // Reset should clear history
        sim.reset();
        assert!(sim.measurement_history().is_empty());
        assert_eq!(sim.measurement_count(), 0);
    }

    #[test]
    fn test_display_format() {
        // Test deterministic 0: m0=0
        let r = SymbolicMeasurementResult {
            outcome: BitSet::new(),
            flip: false,
            is_deterministic: true,
            index: 0,
        };
        assert_eq!(format!("{r}"), "m0=0");

        // Test deterministic 1: m1=1
        let r = SymbolicMeasurementResult {
            outcome: BitSet::new(),
            flip: true,
            is_deterministic: true,
            index: 1,
        };
        assert_eq!(format!("{r}"), "m1=1");

        // Test single dependency, no flip: m2=m0
        let r = SymbolicMeasurementResult {
            outcome: BitSet::single(0),
            flip: false,
            is_deterministic: true,
            index: 2,
        };
        assert_eq!(format!("{r}"), "m2=m0");

        // Test single dependency with flip: m2=m0^1
        let r = SymbolicMeasurementResult {
            outcome: BitSet::single(0),
            flip: true,
            is_deterministic: true,
            index: 2,
        };
        assert_eq!(format!("{r}"), "m2=m0^1");

        // Test multiple dependencies, no flip (largest to smallest): m5=m3^m1
        let r = SymbolicMeasurementResult {
            outcome: [1, 3].into_iter().collect(),
            flip: false,
            is_deterministic: true,
            index: 5,
        };
        assert_eq!(format!("{r}"), "m5=m3^m1");

        // Test multiple dependencies with flip: m5=m3^m1^1
        let r = SymbolicMeasurementResult {
            outcome: [1, 3].into_iter().collect(),
            flip: true,
            is_deterministic: true,
            index: 5,
        };
        assert_eq!(format!("{r}"), "m5=m3^m1^1");

        // Test non-deterministic: m0=?
        let r = SymbolicMeasurementResult {
            outcome: BitSet::single(0),
            flip: false,
            is_deterministic: false,
            index: 0,
        };
        assert_eq!(format!("{r}"), "m0=?");
    }

    #[test]
    fn test_display_in_simulation() {
        let mut sim = SymbolicSparseStabVecSet::new(2);

        // Deterministic 0
        let r = sim.mz(&[0])[0].clone();
        assert_eq!(format!("{r}"), "m0=0");

        sim.reset();

        // Apply X for deterministic 1
        sim.x(&[0]);
        let r = sim.mz(&[0])[0].clone();
        assert_eq!(format!("{r}"), "m0=1");

        sim.reset();

        // Bell state
        sim.h(&[0]).cx(&[(0, 1)]);
        let r0 = sim.mz(&[0])[0].clone();
        let r1 = sim.mz(&[1])[0].clone();

        // r0 is non-deterministic
        assert_eq!(format!("{r0}"), "m0=?");
        // r1 is deterministic, depends on m0
        assert_eq!(format!("{r1}"), "m1=m0");
    }

    #[test]
    fn test_history_formatting() {
        let mut sim = SymbolicSparseStabVecSet::new(3);

        // Create GHZ state: first measurement non-deterministic, rest deterministic
        sim.h(&[0]).cx(&[(0, 1)]).cx(&[(1, 2)]);
        sim.mz(&[0]); // non-det
        sim.mz(&[1]); // det
        sim.mz(&[2]); // det

        let history = sim.measurement_history();

        // Test format_all (also tests Display)
        assert_eq!(history.format_all(), "[m0=?, m1=m0, m2=m0]");
        assert_eq!(format!("{history}"), "[m0=?, m1=m0, m2=m0]");

        // Test format_deterministic
        assert_eq!(history.format_deterministic(), "[m1=m0, m2=m0]");

        // Test format_nondeterministic
        assert_eq!(history.format_nondeterministic(), "[m0=?]");

        // Test Debug format shows struct fields
        let r = &history[0];
        let debug_str = format!("{r:?}");
        assert!(debug_str.contains("outcome"));
        assert!(debug_str.contains("flip"));
        assert!(debug_str.contains("is_deterministic"));
        assert!(debug_str.contains("index"));
    }

    #[test]
    fn test_history_formatting_empty() {
        let sim = SymbolicSparseStabVecSet::new(2);
        let history = sim.measurement_history();

        assert_eq!(history.format_all(), "[]");
        assert_eq!(history.format_deterministic(), "[]");
        assert_eq!(history.format_nondeterministic(), "[]");
        assert_eq!(format!("{history}"), "[]");
    }

    #[test]
    fn test_history_formatting_with_flips() {
        let mut sim = SymbolicSparseStabVecSet::new(2);

        // Apply X to get flip=1, then measure
        sim.x(&[0]);
        sim.mz(&[0]); // det with flip=1
        sim.mz(&[1]); // det with flip=0

        let history = sim.measurement_history();
        assert_eq!(history.format_all(), "[m0=1, m1=0]");
        assert_eq!(history.format_deterministic(), "[m0=1, m1=0]");
        assert_eq!(history.format_nondeterministic(), "[]");
    }

    /// Two-round X-stabilizer check: m1 must depend on m0 across reset.
    #[test]
    fn test_pz_two_round_x_check() {
        use crate::measurement_sampler::MeasurementKind;
        let mut sim = SymbolicSparseStabVecSet::new(3);

        // Round 1: measure X₁X₂ via ancilla q0
        sim.h(&[0]).cx(&[(0, 1)]).cx(&[(0, 2)]).h(&[0]);
        let r0 = sim.mz(&[0]);
        assert!(!r0[0].is_deterministic, "m0 should be non-det");

        sim.pz(0);

        // Round 2: same stabilizer
        sim.h(&[0]).cx(&[(0, 1)]).cx(&[(0, 2)]).h(&[0]);
        let r1 = sim.mz(&[0]);

        assert!(r1[0].is_deterministic, "m1 should be det, got: {}", r1[0]);
        assert_eq!(format!("{}", r1[0]), "m1=m0");

        let kinds = MeasurementKind::from_history(sim.measurement_history());
        assert!(matches!(kinds[0], MeasurementKind::Random));
        assert!(matches!(kinds[1], MeasurementKind::Copy(0)));
    }

    /// Multi-round X-check: correlations must survive 6 reset cycles.
    #[test]
    fn test_pz_six_round_x_check() {
        use crate::measurement_sampler::MeasurementKind;
        let mut sim = SymbolicSparseStabVecSet::new(3);

        for _ in 0..6 {
            sim.h(&[0]).cx(&[(0, 1)]).cx(&[(0, 2)]).h(&[0]);
            sim.mz(&[0]);
            sim.pz(0);
        }

        let kinds = MeasurementKind::from_history(sim.measurement_history());
        assert_eq!(kinds.len(), 6);
        assert!(
            matches!(kinds[0], MeasurementKind::Random),
            "m0: {:?}",
            kinds[0]
        );
        // All subsequent measurements must depend on a prior measurement
        for (i, k) in kinds.iter().enumerate().skip(1) {
            assert!(
                matches!(k, MeasurementKind::Copy(_)),
                "m{i} should be Copy, got {k:?}"
            );
        }
    }

    /// After PZ, the reset qubit itself must be fresh (deterministic |0⟩).
    #[test]
    fn test_pz_reset_qubit_is_fresh() {
        let mut sim = SymbolicSparseStabVecSet::new(2);

        // Entangle q0 and q1 via Bell state
        sim.h(&[0]).cx(&[(0, 1)]);
        // Measure q0 (non-det)
        let r = sim.mz(&[0]);
        assert!(!r[0].is_deterministic);

        // Reset q0
        sim.pz(0);

        // Measuring q0 again must give deterministic 0 (fresh |0⟩)
        let r2 = sim.mz(&[0]);
        assert!(r2[0].is_deterministic, "reset qubit should be det");
        assert_eq!(format!("{}", r2[0]), "m1=0", "reset qubit should measure 0");
    }

    /// PZ on a qubit that was never entangled should be a no-op.
    #[test]
    fn test_pz_on_fresh_qubit() {
        let mut sim = SymbolicSparseStabVecSet::new(2);

        // PZ on |0⟩ should not change anything
        sim.pz(0);

        let r = sim.mz(&[0]);
        assert!(r[0].is_deterministic);
        assert_eq!(format!("{}", r[0]), "m0=0");
    }
}
