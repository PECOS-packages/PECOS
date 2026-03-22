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

//! Symbolic stabilizer simulator with measurement-indexed signs.
//!
//! This module provides [`SymbolicSparseStab`], the default BitSet-based stabilizer simulator
//! that tracks measurement dependencies rather than collapsing to concrete outcomes.
//!
//! Uses `BitSet` for O(1) toggle operations, making it significantly faster than
//! [`SymbolicSparseStabVecSet`](crate::symbolic_sparse_stab::SymbolicSparseStabVecSet)
//! for circuits with 100+ qubits.

use crate::QuantumSimulator;
use crate::sign_algebra::{SignAlgebra, SymbolicSign};
use crate::symbolic_gens::SymbolicGensBitSet;
use crate::symbolic_sparse_stab::{MeasurementHistory, SymbolicMeasurementResult};
use pecos_core::BitSet;
use std::mem;

/// Symbolic stabilizer simulator that tracks measurement dependencies.
///
/// This is the default implementation using `BitSet` for O(1) toggle operations,
/// making it significantly faster than [`SymbolicSparseStabVecSet`](crate::symbolic_sparse_stab::SymbolicSparseStabVecSet)
/// for circuits with 100+ qubits.
///
/// # Example
/// ```rust
/// use pecos_simulators::SymbolicSparseStab;
///
/// let mut sim = SymbolicSparseStab::new(2);
/// sim.h(0).cx(0, 1);  // Create Bell state
/// let r0 = sim.mz(0); // Non-deterministic
/// let r1 = sim.mz(1); // Deterministic, depends on r0
/// assert_eq!(r0.outcome, r1.outcome);
/// ```
#[derive(Clone, Debug)]
pub struct SymbolicSparseStab {
    num_qubits: usize,
    stabs: SymbolicGensBitSet,
    destabs: SymbolicGensBitSet,
    measurement_counter: usize,
    measurement_history: MeasurementHistory,
}

impl SymbolicSparseStab {
    /// Create a new BitSet-based symbolic stabilizer simulator.
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let mut sim = Self {
            num_qubits,
            stabs: SymbolicGensBitSet::new(num_qubits),
            destabs: SymbolicGensBitSet::new(num_qubits),
            measurement_counter: 0,
            measurement_history: MeasurementHistory::new(),
        };
        sim.stabs.init_all_z();
        sim.destabs.init_all_x();
        sim
    }

    /// Get the number of qubits.
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
    #[inline]
    #[must_use]
    pub fn measurement_history(&self) -> &MeasurementHistory {
        &self.measurement_history
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

    /// Pauli X gate. X -> X, Z -> -Z
    #[inline]
    pub fn x(&mut self, q: usize) -> &mut Self {
        self.stabs.signs_minus ^= &self.stabs.col_z[q];
        self
    }

    /// Pauli Y gate. X -> -X, Z -> -Z
    #[inline]
    pub fn y(&mut self, q: usize) -> &mut Self {
        // XOR elements in symmetric difference of col_x[q] and col_z[q] into signs_minus
        let sym_diff = &self.stabs.col_x[q] ^ &self.stabs.col_z[q];
        self.stabs.signs_minus ^= &sym_diff;
        self
    }

    /// Pauli Z gate. X -> -X, Z -> Z
    #[inline]
    pub fn z(&mut self, q: usize) -> &mut Self {
        self.stabs.signs_minus ^= &self.stabs.col_x[q];
        self
    }

    /// Sqrt of Z gate (S gate). X -> Y, Z -> Z
    #[inline]
    pub fn sz(&mut self, q: usize) -> &mut Self {
        // X -> i: track phase changes
        // i * i = -1, so if already has i, add minus and remove i
        // Compute intersection manually using BitSet words
        let intersection = {
            let mut result = BitSet::new();
            for i in &self.stabs.signs_i {
                if self.stabs.col_x[q].contains(i) {
                    result.insert(i);
                }
            }
            result
        };
        self.stabs.signs_minus ^= &intersection;
        self.stabs.signs_i ^= &self.stabs.col_x[q];

        // Update the Pauli structure (X -> Y means add Z component)
        // Need to collect indices first to avoid borrow issues
        let col_x_indices: Vec<usize> = self.stabs.col_x[q].iter().collect();

        // Update stabs
        self.stabs.col_z[q] ^= &self.stabs.col_x[q];
        let q_set = BitSet::single(q);
        for i in &col_x_indices {
            self.stabs.row_z[*i] ^= &q_set;
        }

        // Update destabs
        let destab_col_x_indices: Vec<usize> = self.destabs.col_x[q].iter().collect();
        self.destabs.col_z[q] ^= &self.destabs.col_x[q];
        for i in &destab_col_x_indices {
            self.destabs.row_z[*i] ^= &q_set;
        }

        self
    }

    /// Hadamard gate. X -> Z, Z -> X, Y -> -Y
    #[inline]
    pub fn h(&mut self, q: usize) -> &mut Self {
        // Y -> -Y: add minus for generators that have both X and Z on this qubit
        // Compute intersection
        let intersection = {
            let mut result = BitSet::new();
            for i in &self.stabs.col_x[q] {
                if self.stabs.col_z[q].contains(i) {
                    result.insert(i);
                }
            }
            result
        };
        self.stabs.signs_minus ^= &intersection;

        // Swap X and Z for this qubit - process stabs
        {
            // Elements only in col_x (not in col_z)
            let only_x: Vec<usize> = self.stabs.col_x[q]
                .iter()
                .filter(|i| !self.stabs.col_z[q].contains(*i))
                .collect();
            // Elements only in col_z (not in col_x)
            let only_z: Vec<usize> = self.stabs.col_z[q]
                .iter()
                .filter(|i| !self.stabs.col_x[q].contains(*i))
                .collect();

            for i in only_x {
                self.stabs.row_x[i].remove(q);
                self.stabs.row_z[i].insert(q);
            }
            for i in only_z {
                self.stabs.row_z[i].remove(q);
                self.stabs.row_x[i].insert(q);
            }
            mem::swap(&mut self.stabs.col_x[q], &mut self.stabs.col_z[q]);
        }

        // Swap X and Z for destabs
        {
            let only_x: Vec<usize> = self.destabs.col_x[q]
                .iter()
                .filter(|i| !self.destabs.col_z[q].contains(*i))
                .collect();
            let only_z: Vec<usize> = self.destabs.col_z[q]
                .iter()
                .filter(|i| !self.destabs.col_x[q].contains(*i))
                .collect();

            for i in only_x {
                self.destabs.row_x[i].remove(q);
                self.destabs.row_z[i].insert(q);
            }
            for i in only_z {
                self.destabs.row_z[i].remove(q);
                self.destabs.row_x[i].insert(q);
            }
            mem::swap(&mut self.destabs.col_x[q], &mut self.destabs.col_z[q]);
        }

        self
    }

    /// CNOT gate. IX -> IX, XI -> XX, IZ -> ZZ, ZI -> ZI
    #[inline]
    pub fn cx(&mut self, q1: usize, q2: usize) -> &mut Self {
        // Pre-create the single-element BitSets for toggling
        let q1_set = BitSet::single(q1);
        let q2_set = BitSet::single(q2);

        // Process stabs
        {
            // Handle col_x: XI -> XX
            let x_col_indices: Vec<usize> = self.stabs.col_x[q1].iter().collect();
            for i in &x_col_indices {
                self.stabs.row_x[*i] ^= &q2_set;
            }
            let x_col_copy = self.stabs.col_x[q1].clone();
            self.stabs.col_x[q2] ^= &x_col_copy;

            // Handle col_z: IZ -> ZZ
            let z_col_indices: Vec<usize> = self.stabs.col_z[q2].iter().collect();
            for i in &z_col_indices {
                self.stabs.row_z[*i] ^= &q1_set;
            }
            let z_col_copy = self.stabs.col_z[q2].clone();
            self.stabs.col_z[q1] ^= &z_col_copy;
        }

        // Process destabs
        {
            // Handle col_x: XI -> XX
            let x_col_indices: Vec<usize> = self.destabs.col_x[q1].iter().collect();
            for i in &x_col_indices {
                self.destabs.row_x[*i] ^= &q2_set;
            }
            let x_col_copy = self.destabs.col_x[q1].clone();
            self.destabs.col_x[q2] ^= &x_col_copy;

            // Handle col_z: IZ -> ZZ
            let z_col_indices: Vec<usize> = self.destabs.col_z[q2].iter().collect();
            for i in &z_col_indices {
                self.destabs.row_z[*i] ^= &q1_set;
            }
            let z_col_copy = self.destabs.col_z[q2].clone();
            self.destabs.col_z[q1] ^= &z_col_copy;
        }

        self
    }

    // ==================== Measurement ====================

    /// Measure a qubit in the Z basis.
    #[inline]
    pub fn mz(&mut self, q: usize) -> SymbolicMeasurementResult {
        let result = if self.stabs.col_x[q].is_empty() {
            self.deterministic_meas(q)
        } else {
            self.nondeterministic_meas(q)
        };

        self.measurement_history.push(result.clone());
        result
    }

    /// Handle a deterministic measurement.
    fn deterministic_meas(&mut self, q: usize) -> SymbolicMeasurementResult {
        let index = self.measurement_counter;
        self.measurement_counter += 1;

        // Count minuses from destabilizers that anti-commute with Z_q
        let mut num_minuses = 0;
        for i in &self.destabs.col_x[q] {
            if self.stabs.signs_minus.contains(i) {
                num_minuses += 1;
            }
        }

        let mut num_is = 0;
        for i in &self.destabs.col_x[q] {
            if self.stabs.signs_i.contains(i) {
                num_is += 1;
            }
        }

        // Account for Pauli multiplication phases
        let mut cumulative_x = BitSet::new();
        for row in &self.destabs.col_x[q] {
            // Count intersection of row_z[row] and cumulative_x
            for z in &self.stabs.row_z[row] {
                if cumulative_x.contains(z) {
                    num_minuses += 1;
                }
            }
            cumulative_x ^= &self.stabs.row_x[row];
        }

        if num_is & 3 != 0 {
            num_minuses += 1;
        }

        let flip = num_minuses & 1 != 0;

        // XOR together the symbolic signs
        let mut result_sign = SymbolicSign::empty();
        for row in &self.destabs.col_x[q] {
            result_sign.multiply_assign(&self.stabs.signs[row]);
        }

        SymbolicMeasurementResult {
            outcome: result_sign.measurements,
            flip,
            is_deterministic: true,
            index,
        }
    }

    /// Handle a non-deterministic measurement.
    #[allow(clippy::too_many_lines)]
    fn nondeterministic_meas(&mut self, q: usize) -> SymbolicMeasurementResult {
        let measurement_index = self.measurement_counter;
        self.measurement_counter += 1;

        let mut anticom_stabs_col = self.stabs.col_x[q].clone();
        let mut anticom_destabs_col = self.destabs.col_x[q].clone();

        // Find a stabilizer to replace (choose smallest weight)
        let mut smallest_wt = 2 * self.num_qubits + 2;
        let mut removed_id: Option<usize> = None;

        for stab_id in &anticom_stabs_col {
            let weight = self.stabs.row_x[stab_id].len() + self.stabs.row_z[stab_id].len();
            if weight < smallest_wt {
                smallest_wt = weight;
                removed_id = Some(stab_id);
            }
        }

        let id = removed_id.expect("Critical error: removed_id was None");
        anticom_stabs_col.remove(id);
        let removed_row_x = self.stabs.row_x[id].clone();
        let removed_row_z = self.stabs.row_z[id].clone();

        // Phase tracking for anticommuting stabilizers
        if self.stabs.signs_minus.contains(id) {
            self.stabs.signs_minus ^= &anticom_stabs_col;
        }

        // Handle imaginary component propagation
        if self.stabs.signs_i.contains(id) {
            self.stabs.signs_i.remove(id);

            let gens_common: Vec<usize> = self
                .stabs
                .signs_i
                .iter()
                .filter(|i| anticom_stabs_col.contains(*i))
                .collect();
            let gens_only_stabs: Vec<usize> = anticom_stabs_col
                .iter()
                .filter(|i| !self.stabs.signs_i.contains(*i))
                .collect();

            for i in gens_common {
                let i_set = BitSet::single(i);
                self.stabs.signs_minus ^= &i_set;
                self.stabs.signs_i.remove(i);
            }

            for i in gens_only_stabs {
                self.stabs.signs_i.insert(i);
            }
        }

        // Multiply all other anticommuting stabilizers by the removed one
        let removed_sign = self.stabs.signs[id].clone();
        for g in &anticom_stabs_col {
            self.stabs.signs[g].multiply_assign(&removed_sign);

            // Count intersection for phase
            let mut num_minuses = 0;
            for z in &removed_row_z {
                if self.stabs.row_x[g].contains(z) {
                    num_minuses += 1;
                }
            }
            if num_minuses & 1 != 0 {
                let g_set = BitSet::single(g);
                self.stabs.signs_minus ^= &g_set;
            }

            // Update Pauli structure
            self.stabs.row_x[g] ^= &removed_row_x;
            self.stabs.row_z[g] ^= &removed_row_z;
        }

        // Update column storage for stabilizers
        for i in &removed_row_x {
            self.stabs.col_x[i] ^= &anticom_stabs_col;
        }
        for i in &removed_row_z {
            self.stabs.col_z[i] ^= &anticom_stabs_col;
        }

        // Remove the old stabilizer
        for i in &self.stabs.row_x[id] {
            self.stabs.col_x[i].remove(id);
        }
        for i in &self.stabs.row_z[id] {
            self.stabs.col_z[i].remove(id);
        }

        // Replace with the measured stabilizer Z_q
        self.stabs.col_z[q].insert(id);
        self.stabs.row_x[id].clear();
        self.stabs.row_z[id].clear();
        self.stabs.row_z[id].insert(q);

        // Set the sign of the new stabilizer
        self.stabs.signs[id] = SymbolicSign::single(measurement_index);
        self.stabs.signs_minus.remove(id);
        self.stabs.signs_i.remove(id);

        // Update destabilizers
        for i in &self.destabs.row_x[id] {
            self.destabs.col_x[i].remove(id);
        }
        for i in &self.destabs.row_z[id] {
            self.destabs.col_z[i].remove(id);
        }

        anticom_destabs_col.remove(id);

        for i in &removed_row_x {
            self.destabs.col_x[i].insert(id);
            self.destabs.col_x[i] ^= &anticom_destabs_col;
        }
        for i in &removed_row_z {
            self.destabs.col_z[i].insert(id);
            self.destabs.col_z[i] ^= &anticom_destabs_col;
        }

        for row in &anticom_destabs_col {
            self.destabs.row_x[row] ^= &removed_row_x;
            self.destabs.row_z[row] ^= &removed_row_z;
        }

        self.destabs.row_x[id] = removed_row_x;
        self.destabs.row_z[id] = removed_row_z;

        SymbolicMeasurementResult {
            outcome: BitSet::single(measurement_index),
            flip: false,
            is_deterministic: false,
            index: measurement_index,
        }
    }
}

impl QuantumSimulator for SymbolicSparseStab {
    #[inline]
    fn reset(&mut self) -> &mut Self {
        Self::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bell_state_bitset() {
        let mut sim = SymbolicSparseStab::new(2);

        // Create Bell state
        sim.h(0).cx(0, 1);

        // Measure qubit 0 - should be non-deterministic
        let r0 = sim.mz(0);
        assert!(!r0.is_deterministic);
        assert_eq!(r0.outcome.len(), 1);
        assert!(r0.outcome.contains(0));
        assert_eq!(r0.index, 0);

        // Measure qubit 1 - should be deterministic
        let r1 = sim.mz(1);
        assert!(r1.is_deterministic);
        assert_eq!(r0.outcome, r1.outcome);
        assert_eq!(r1.index, 1);
    }

    #[test]
    fn test_product_state_bitset() {
        let mut sim = SymbolicSparseStab::new(2);

        let r0 = sim.mz(0);
        assert!(r0.is_deterministic);
        assert!(r0.outcome.is_empty());
        assert_eq!(r0.index, 0);

        let r1 = sim.mz(1);
        assert!(r1.is_deterministic);
        assert!(r1.outcome.is_empty());
        assert_eq!(r1.index, 1);
    }
}
