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

//! Symbolic stabilizer simulator with measurement-indexed signs.
//!
//! This module provides [`SymbolicSparseStab`], a stabilizer simulator that tracks
//! measurement dependencies rather than collapsing to concrete outcomes.
//!
//! Instead of randomly choosing 0 or 1 for non-deterministic measurements, this simulator
//! assigns each measurement a unique index and tracks which measurements contribute to
//! each stabilizer's sign via XOR (symmetric difference).
//!
//! Every measurement receives a unique index. Use the `is_deterministic` field in the
//! result to distinguish deterministic from non-deterministic measurements if needed.

use crate::QuantumSimulator;
use crate::sign_algebra::{SignAlgebra, SymbolicSign};
use crate::symbolic_gens::SymbolicGens;
use core::mem;
use pecos_core::{IndexableElement, Set, VecSet};
use std::collections::BTreeSet;

/// Standard type alias for symbolic sparse stabilizer simulator.
pub type StdSymbolicSparseStab = SymbolicSparseStab<VecSet<usize>, usize>;

/// Result of a symbolic measurement.
///
/// Instead of a concrete 0/1 outcome, this contains the set of measurement indices
/// whose outcomes XOR together to determine this measurement's result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicMeasurementResult {
    /// The set of measurement indices whose outcomes XOR to give this result.
    /// Empty set means the result is deterministically 0.
    /// A set containing just this measurement's index means it's non-deterministic.
    pub outcome: BTreeSet<usize>,
    /// Whether this measurement was deterministic (outcome determined by prior measurements).
    pub is_deterministic: bool,
    /// The index assigned to this measurement.
    pub index: usize,
}

/// A symbolic stabilizer simulator that tracks measurement dependencies.
///
/// This simulator is based on the same stabilizer/destabilizer formalism as [`SparseStab`],
/// but instead of collapsing measurements to concrete outcomes, it tracks which measurements
/// contribute to each outcome.
///
/// # Type Parameters
/// - `T`: Set type for sparse storage
/// - `E`: Element type (qubit index type)
///
/// # Use Cases
/// - Analyzing measurement dependency graphs
/// - Understanding which measurements affect which outcomes
/// - Pauli frame tracking / deferred measurement patterns
/// - Verifying measurement patterns in quantum error correction
///
/// # Example
/// ```rust
/// use pecos_qsim::symbolic_sparse_stab::StdSymbolicSparseStab;
/// use pecos_qsim::QuantumSimulator;
///
/// let mut sim = StdSymbolicSparseStab::new(2);
///
/// // Create Bell state
/// sim.h(0).cx(0, 1);
///
/// // Measure both qubits
/// let r0 = sim.mz(0);  // Non-deterministic: outcome depends on measurement 0
/// let r1 = sim.mz(1);  // Deterministic: outcome equals measurement 0's outcome
///
/// // r0.outcome = {0} (depends on measurement 0)
/// // r1.outcome = {0} (also depends on measurement 0, showing correlation)
/// assert!(!r0.is_deterministic);
/// assert!(r1.is_deterministic);
/// assert_eq!(r0.outcome, r1.outcome);  // Same dependency = correlated
/// ```
#[derive(Clone, Debug)]
pub struct SymbolicSparseStab<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    num_qubits: usize,
    stabs: SymbolicGens<T, E>,
    destabs: SymbolicGens<T, E>,
    /// Counter for assigning unique indices to measurements
    measurement_counter: usize,
}

impl<T, E> SymbolicSparseStab<T, E>
where
    E: IndexableElement,
    T: for<'a> Set<'a, Element = E>,
{
    /// Create a new symbolic stabilizer simulator.
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let mut sim = Self {
            num_qubits,
            stabs: SymbolicGens::<T, E>::new(num_qubits),
            destabs: SymbolicGens::<T, E>::new(num_qubits),
            measurement_counter: 0,
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
    fn tableau_string(num_qubits: usize, gens: &SymbolicGens<T, E>) -> String {
        use std::fmt::Write;

        let mut result = String::new();
        for i in 0..num_qubits {
            // Format the symbolic sign
            let sign = &gens.signs[i];
            if sign.measurements.is_empty() {
                result.push_str("{} ");
            } else {
                let indices: Vec<String> = sign
                    .measurements
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect();
                let _ = write!(result, "{{{}}} ", indices.join(","));
            }

            // Format the Pauli string
            for qubit in 0..num_qubits {
                let qubit_e = E::from_index(qubit);
                let in_row_x = gens.row_x[i].contains(&qubit_e);
                let in_row_z = gens.row_z[i].contains(&qubit_e);

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
        self
    }

    // ==================== Gate Operations ====================
    // These are largely the same as SparseStab, but without sign updates
    // (symbolic signs don't change under Clifford gates - only under measurement)

    /// Pauli X gate. X -> X, Z -> -Z
    /// For symbolic simulation, we don't track the phase flip.
    #[inline]
    pub fn x(&mut self, _q: E) -> &mut Self {
        // In symbolic simulation, X gate doesn't change the measurement dependencies
        // The phase flip on Z doesn't matter because we only track measurement indices
        self
    }

    /// Pauli Y gate. X -> -X, Z -> -Z
    #[inline]
    pub fn y(&mut self, _q: E) -> &mut Self {
        // Same as X - phase flips don't affect symbolic signs
        self
    }

    /// Pauli Z gate. X -> -X, Z -> Z
    #[inline]
    pub fn z(&mut self, _q: E) -> &mut Self {
        // Same as X - phase flips don't affect symbolic signs
        self
    }

    /// Sqrt of Z gate (S gate).
    #[inline]
    pub fn sz(&mut self, q: E) -> &mut Self {
        let qu = q.to_index();

        // Update the Pauli structure (X -> Y, Z -> Z)
        // but ignore the phase changes for symbolic simulation
        for g in [&mut self.stabs, &mut self.destabs] {
            g.col_z[qu] ^= &g.col_x[qu];

            for &i in g.col_x[qu].iter() {
                let iu = i.to_index();
                g.row_z[iu] ^= &q;
            }
        }
        self
    }

    /// Hadamard gate. X -> Z, Z -> X
    #[inline]
    pub fn h(&mut self, q: E) -> &mut Self {
        let qu = q.to_index();

        // Swap X and Z for this qubit (no phase tracking needed)
        for g in [&mut self.stabs, &mut self.destabs] {
            for i in g.col_x[qu].difference(&g.col_z[qu]) {
                let iu = i.to_index();
                g.row_x[iu].remove(&q);
                g.row_z[iu].insert(q);
            }

            for i in g.col_z[qu].difference(&g.col_x[qu]) {
                let iu = i.to_index();
                g.row_z[iu].remove(&q);
                g.row_x[iu].insert(q);
            }

            mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
        }
        self
    }

    /// CNOT gate. IX -> IX, XI -> XX, IZ -> ZZ, ZI -> ZI
    #[inline]
    pub fn cx(&mut self, q1: E, q2: E) -> &mut Self {
        let qu1 = q1.to_index();
        let qu2 = q2.to_index();

        for g in &mut [&mut self.stabs, &mut self.destabs] {
            let (qu_min, qu_max) = if qu1 < qu2 { (qu1, qu2) } else { (qu2, qu1) };

            // Handle col_x: XI -> XX
            {
                let (_left, right) = g.col_x.split_at_mut(qu_min);
                let (mid, right) = right.split_at_mut(qu_max - qu_min);
                let col_x_min = &mut mid[0];
                let col_x_max = &mut right[0];

                let (col_x_qu1, col_x_qu2) = if qu1 < qu2 {
                    (col_x_min, col_x_max)
                } else {
                    (col_x_max, col_x_min)
                };

                let mut q2_set = T::new();
                q2_set.insert(q2);

                for i in col_x_qu1.iter() {
                    let iu = i.to_index();
                    g.row_x[iu].symmetric_difference_update(&q2_set);
                }
                col_x_qu2.symmetric_difference_update(col_x_qu1);
            }

            // Handle col_z: IZ -> ZZ
            {
                let (_left, right) = g.col_z.split_at_mut(qu_min);
                let (mid, right) = right.split_at_mut(qu_max - qu_min);
                let col_z_min = &mut mid[0];
                let col_z_max = &mut right[0];

                let (col_z_qu1, col_z_qu2) = if qu1 < qu2 {
                    (col_z_min, col_z_max)
                } else {
                    (col_z_max, col_z_min)
                };

                let mut q1_set = T::new();
                q1_set.insert(q1);

                for i in col_z_qu2.iter() {
                    let iu = i.to_index();
                    g.row_z[iu].symmetric_difference_update(&q1_set);
                }
                col_z_qu1.symmetric_difference_update(col_z_qu2);
            }
        }
        self
    }

    // ==================== Measurement ====================

    /// Measure a qubit in the Z basis.
    ///
    /// Returns a [`SymbolicMeasurementResult`] containing the set of measurement indices
    /// whose outcomes XOR together to determine this measurement's result.
    #[inline]
    pub fn mz(&mut self, q: E) -> SymbolicMeasurementResult {
        let qu = q.to_index();

        if self.stabs.col_x[qu].is_empty() {
            // Deterministic measurement
            self.deterministic_meas(q)
        } else {
            // Non-deterministic measurement
            self.nondeterministic_meas(q)
        }
    }

    /// Handle a deterministic measurement.
    /// The outcome is determined by `XORing` the signs of destabilizers that have X on this qubit.
    fn deterministic_meas(&mut self, q: E) -> SymbolicMeasurementResult {
        let qu = q.to_index();

        // Assign index and increment counter
        let index = self.measurement_counter;
        self.measurement_counter += 1;

        // XOR together the signs of all destabilizers that have X on qubit q
        // These are the destabilizers that anti-commute with Z_q
        let mut result_sign = SymbolicSign::empty();

        for row in self.destabs.col_x[qu].iter() {
            let rowu = row.to_index();
            result_sign.multiply_assign(&self.stabs.signs[rowu]);
        }

        SymbolicMeasurementResult {
            outcome: result_sign.measurements,
            is_deterministic: true,
            index,
        }
    }

    /// Handle a non-deterministic measurement.
    /// Assigns a new measurement index and updates the stabilizer tableau.
    #[allow(clippy::too_many_lines)]
    fn nondeterministic_meas(&mut self, q: E) -> SymbolicMeasurementResult {
        let qu = q.to_index();

        // Non-deterministic measurements always get an index (required for tracking)
        let measurement_index = self.measurement_counter;
        self.measurement_counter += 1;

        let mut anticom_stabs_col = self.stabs.col_x[qu].clone();
        let mut anticom_destabs_col = self.destabs.col_x[qu].clone();

        // Find a stabilizer to replace (choose smallest weight for efficiency)
        let mut smallest_wt = 2 * self.num_qubits + 2;
        let mut removed_id: Option<E> = None;

        for stab_id in anticom_stabs_col.iter() {
            let stab_usize = stab_id.to_index();
            let weight = self.stabs.row_x[stab_usize].len() + self.stabs.row_z[stab_usize].len();

            if weight < smallest_wt {
                smallest_wt = weight;
                removed_id = Some(*stab_id);
            }
        }

        let id = removed_id.expect("Critical error: removed_id was None");
        anticom_stabs_col.remove(&id);
        let id_usize = id.to_index();
        let removed_row_x = self.stabs.row_x[id_usize].clone();
        let removed_row_z = self.stabs.row_z[id_usize].clone();

        // Multiply all other anticommuting stabilizers by the removed one
        // This includes multiplying their signs (XOR of measurement indices)
        let removed_sign = self.stabs.signs[id_usize].clone();
        for g in anticom_stabs_col.iter() {
            let gen_usize = g.to_index();

            // Multiply the signs
            self.stabs.signs[gen_usize].multiply_assign(&removed_sign);

            // Update the Pauli structure
            self.stabs.row_x[gen_usize] ^= &removed_row_x;
            self.stabs.row_z[gen_usize] ^= &removed_row_z;
        }

        // Update column storage for stabilizers
        for i in removed_row_x.iter() {
            let iu = i.to_index();
            self.stabs.col_x[iu] ^= &anticom_stabs_col;
        }

        for i in removed_row_z.iter() {
            let iu = i.to_index();
            self.stabs.col_z[iu] ^= &anticom_stabs_col;
        }

        // Remove the old stabilizer
        for i in self.stabs.row_x[id_usize].iter() {
            let iu = i.to_index();
            self.stabs.col_x[iu].remove(&id);
        }

        for i in self.stabs.row_z[id_usize].iter() {
            let iu = i.to_index();
            self.stabs.col_z[iu].remove(&id);
        }

        // Replace with the measured stabilizer Z_q
        self.stabs.col_z[qu].insert(id);
        self.stabs.row_x[id_usize].clear();
        self.stabs.row_z[id_usize].clear();
        self.stabs.row_z[id_usize].insert(q);

        // Set the sign of the new stabilizer to this measurement's index
        self.stabs.signs[id_usize] = SymbolicSign::single(measurement_index);

        // Update destabilizers
        for i in self.destabs.row_x[id_usize].iter() {
            let iu = i.to_index();
            self.destabs.col_x[iu].remove(&id);
        }

        for i in self.destabs.row_z[id_usize].iter() {
            let iu = i.to_index();
            self.destabs.col_z[iu].remove(&id);
        }

        anticom_destabs_col.remove(&id);

        for i in removed_row_x.iter() {
            let iu = i.to_index();
            self.destabs.col_x[iu].insert(id);
            self.destabs.col_x[iu] ^= &anticom_destabs_col;
        }

        for i in removed_row_z.iter() {
            let iu = i.to_index();
            self.destabs.col_z[iu].insert(id);
            self.destabs.col_z[iu] ^= &anticom_destabs_col;
        }

        for row in anticom_destabs_col.iter() {
            let ru = row.to_index();
            self.destabs.row_x[ru] ^= &removed_row_x;
            self.destabs.row_z[ru] ^= &removed_row_z;
        }

        self.destabs.row_x[id_usize] = removed_row_x;
        self.destabs.row_z[id_usize] = removed_row_z;

        // The outcome is just this measurement's index
        let mut outcome = BTreeSet::new();
        outcome.insert(measurement_index);

        SymbolicMeasurementResult {
            outcome,
            is_deterministic: false,
            index: measurement_index,
        }
    }
}

impl<T, E> QuantumSimulator for SymbolicSparseStab<T, E>
where
    E: IndexableElement,
    T: for<'a> Set<'a, Element = E>,
{
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
        let mut sim = StdSymbolicSparseStab::new(2);

        // Create Bell state
        sim.h(0).cx(0, 1);

        // Measure qubit 0 - should be non-deterministic
        let r0 = sim.mz(0);
        assert!(!r0.is_deterministic);
        assert_eq!(r0.outcome.len(), 1);
        assert!(r0.outcome.contains(&0)); // First measurement has index 0
        assert_eq!(r0.index, 0);

        // Measure qubit 1 - should be deterministic but still gets an index
        let r1 = sim.mz(1);
        assert!(r1.is_deterministic);
        assert_eq!(r0.outcome, r1.outcome); // Same measurement dependency = correlated
        assert_eq!(r1.index, 1);
    }

    #[test]
    fn test_product_state_symbolic() {
        let mut sim = StdSymbolicSparseStab::new(2);

        // Measure qubit 0 without any gates - should be deterministic |0⟩
        let r0 = sim.mz(0);
        assert!(r0.is_deterministic);
        assert!(r0.outcome.is_empty()); // Empty set = deterministic 0
        assert_eq!(r0.index, 0);

        // Measure qubit 1 - also deterministic |0⟩
        let r1 = sim.mz(1);
        assert!(r1.is_deterministic);
        assert!(r1.outcome.is_empty());
        assert_eq!(r1.index, 1);
    }

    #[test]
    fn test_hadamard_measurement_symbolic() {
        let mut sim = StdSymbolicSparseStab::new(1);

        // Apply H to put in superposition
        sim.h(0);

        // Measure - should be non-deterministic
        let r = sim.mz(0);
        assert!(!r.is_deterministic);
        assert_eq!(r.outcome.len(), 1);
        assert!(r.outcome.contains(&0));
        assert_eq!(r.index, 0);
    }

    #[test]
    fn test_ghz_state_symbolic() {
        let mut sim = StdSymbolicSparseStab::new(3);

        // Create GHZ state: (|000⟩ + |111⟩)/√2
        sim.h(0).cx(0, 1).cx(1, 2);

        // Measure qubit 0 - non-deterministic
        let r0 = sim.mz(0);
        assert!(!r0.is_deterministic);
        assert!(r0.outcome.contains(&0));
        assert_eq!(r0.index, 0);

        // Measure qubit 1 - deterministic, depends on measurement 0
        let r1 = sim.mz(1);
        assert!(r1.is_deterministic);
        assert_eq!(r0.outcome, r1.outcome);
        assert_eq!(r1.index, 1);

        // Measure qubit 2 - deterministic, depends on measurement 0
        let r2 = sim.mz(2);
        assert!(r2.is_deterministic);
        assert_eq!(r0.outcome, r2.outcome);
        assert_eq!(r2.index, 2);
    }

    #[test]
    fn test_multiple_independent_measurements() {
        let mut sim = StdSymbolicSparseStab::new(2);

        // Put both qubits in superposition independently
        sim.h(0).h(1);

        // Measure qubit 0 - non-deterministic, index 0
        let r0 = sim.mz(0);
        assert!(!r0.is_deterministic);
        assert!(r0.outcome.contains(&0));
        assert_eq!(r0.index, 0);

        // Measure qubit 1 - non-deterministic, index 1
        let r1 = sim.mz(1);
        assert!(!r1.is_deterministic);
        assert!(r1.outcome.contains(&1));
        assert_eq!(r1.index, 1);

        // They should have different measurement indices (independent)
        assert_ne!(r0.outcome, r1.outcome);
    }

    #[test]
    fn test_measurement_counter() {
        let mut sim = StdSymbolicSparseStab::new(3);
        assert_eq!(sim.measurement_count(), 0);

        // All deterministic measurements - counter always increments
        sim.mz(0);
        assert_eq!(sim.measurement_count(), 1);

        sim.mz(1);
        assert_eq!(sim.measurement_count(), 2);

        sim.mz(2);
        assert_eq!(sim.measurement_count(), 3);
    }

    #[test]
    fn test_measurement_counter_with_nondet() {
        let mut sim = StdSymbolicSparseStab::new(3);
        assert_eq!(sim.measurement_count(), 0);

        // Make non-deterministic measurements
        sim.h(0).h(1).h(2);

        sim.mz(0);
        assert_eq!(sim.measurement_count(), 1);

        sim.mz(1);
        assert_eq!(sim.measurement_count(), 2);

        sim.mz(2);
        assert_eq!(sim.measurement_count(), 3);
    }

    #[test]
    fn test_deterministic_flag() {
        let mut sim = StdSymbolicSparseStab::new(2);

        // Deterministic measurement on |0⟩
        let r0 = sim.mz(0);
        assert!(r0.is_deterministic);

        // Reset and make non-deterministic
        sim.reset();
        sim.h(0);
        let r1 = sim.mz(0);
        assert!(!r1.is_deterministic);
    }
}
