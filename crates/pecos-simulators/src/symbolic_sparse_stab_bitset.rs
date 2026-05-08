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
/// sim.h(&[0]).cx(&[(0, 1)]);  // Create Bell state
/// let r0 = sim.mz(&[0])[0].clone(); // Non-deterministic
/// let r1 = sim.mz(&[1])[0].clone(); // Deterministic, depends on r0
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

fn xor_intersection_into(a: &BitSet, b: &BitSet, target: &mut BitSet) {
    for i in a {
        if b.contains(i) {
            target.toggle(i);
        }
    }
}

fn mul_i_for(signs_minus: &mut BitSet, signs_i: &mut BitSet, indices: &BitSet) {
    for i in indices {
        if signs_i.contains(i) {
            signs_minus.toggle(i);
            signs_i.remove(i);
        } else {
            signs_i.insert(i);
        }
    }
}

fn mul_minus_i_for(signs_minus: &mut BitSet, signs_i: &mut BitSet, indices: &BitSet) {
    *signs_minus ^= indices;
    mul_i_for(signs_minus, signs_i, indices);
}

fn toggle_col_x(gens: &mut SymbolicGensBitSet, q: usize, affected: &BitSet) {
    let old = gens.col_x[q].clone();
    gens.col_x[q] ^= affected;
    for i in &old {
        if !gens.col_x[q].contains(i) {
            gens.row_x[i].remove(q);
        }
    }
    for i in &gens.col_x[q] {
        if !old.contains(i) {
            gens.row_x[i].insert(q);
        }
    }
}

fn toggle_col_z(gens: &mut SymbolicGensBitSet, q: usize, affected: &BitSet) {
    let old = gens.col_z[q].clone();
    gens.col_z[q] ^= affected;
    for i in &old {
        if !gens.col_z[q].contains(i) {
            gens.row_z[i].remove(q);
        }
    }
    for i in &gens.col_z[q] {
        if !old.contains(i) {
            gens.row_z[i].insert(q);
        }
    }
}

fn swap_xz_on(gens: &mut SymbolicGensBitSet, q: usize) {
    let only_x: Vec<usize> = gens.col_x[q]
        .iter()
        .filter(|i| !gens.col_z[q].contains(*i))
        .collect();
    let only_z: Vec<usize> = gens.col_z[q]
        .iter()
        .filter(|i| !gens.col_x[q].contains(*i))
        .collect();

    for i in only_x {
        gens.row_x[i].remove(q);
        gens.row_z[i].insert(q);
    }
    for i in only_z {
        gens.row_z[i].remove(q);
        gens.row_x[i].insert(q);
    }
    mem::swap(&mut gens.col_x[q], &mut gens.col_z[q]);
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
            // XOR elements in symmetric difference of col_x[q] and col_z[q] into signs_minus
            let sym_diff = &self.stabs.col_x[q] ^ &self.stabs.col_z[q];
            self.stabs.signs_minus ^= &sym_diff;
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
        }

        self
    }

    /// Hadamard gate. X -> Z, Z -> X, Y -> -Y
    #[inline]
    pub fn h(&mut self, qubits: &[usize]) -> &mut Self {
        for &q in qubits {
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
        }

        self
    }

    /// Adjoint sqrt of Z gate. X -> -Y, Z -> Z.
    #[inline]
    pub fn szdg(&mut self, qubits: &[usize]) -> &mut Self {
        for &q in qubits {
            let affected = self.stabs.col_x[q].clone();
            mul_minus_i_for(
                &mut self.stabs.signs_minus,
                &mut self.stabs.signs_i,
                &affected,
            );

            for gens in [&mut self.stabs, &mut self.destabs] {
                let affected = gens.col_x[q].clone();
                toggle_col_z(gens, q, &affected);
            }
        }
        self
    }

    /// Sqrt of X gate. X -> X, Z -> -Y.
    #[inline]
    pub fn sx(&mut self, qubits: &[usize]) -> &mut Self {
        for &q in qubits {
            let affected = self.stabs.col_z[q].clone();
            mul_minus_i_for(
                &mut self.stabs.signs_minus,
                &mut self.stabs.signs_i,
                &affected,
            );

            for gens in [&mut self.stabs, &mut self.destabs] {
                let affected = gens.col_z[q].clone();
                toggle_col_x(gens, q, &affected);
            }
        }
        self
    }

    /// Adjoint sqrt of X gate. X -> X, Z -> Y.
    #[inline]
    pub fn sxdg(&mut self, qubits: &[usize]) -> &mut Self {
        for &q in qubits {
            let affected = self.stabs.col_z[q].clone();
            mul_i_for(
                &mut self.stabs.signs_minus,
                &mut self.stabs.signs_i,
                &affected,
            );

            for gens in [&mut self.stabs, &mut self.destabs] {
                let affected = gens.col_z[q].clone();
                toggle_col_x(gens, q, &affected);
            }
        }
        self
    }

    /// Sqrt of Y gate. X -> -Z, Z -> X.
    #[inline]
    pub fn sy(&mut self, qubits: &[usize]) -> &mut Self {
        for &q in qubits {
            self.stabs.signs_minus ^= &self.stabs.col_x[q];
            xor_intersection_into(
                &self.stabs.col_x[q],
                &self.stabs.col_z[q],
                &mut self.stabs.signs_minus,
            );

            for gens in [&mut self.stabs, &mut self.destabs] {
                swap_xz_on(gens, q);
            }
        }
        self
    }

    /// Adjoint sqrt of Y gate. X -> Z, Z -> -X.
    #[inline]
    pub fn sydg(&mut self, qubits: &[usize]) -> &mut Self {
        for &q in qubits {
            self.stabs.signs_minus ^= &self.stabs.col_z[q];
            xor_intersection_into(
                &self.stabs.col_x[q],
                &self.stabs.col_z[q],
                &mut self.stabs.signs_minus,
            );

            for gens in [&mut self.stabs, &mut self.destabs] {
                swap_xz_on(gens, q);
            }
        }
        self
    }

    /// CNOT gate. IX -> IX, XI -> XX, IZ -> ZZ, ZI -> ZI
    #[inline]
    pub fn cx(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        for &(q1, q2) in pairs {
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
        }

        self
    }

    /// Controlled-Y gate. XI -> XY, IX -> ZX, ZI -> ZI, IZ -> ZZ.
    #[inline]
    pub fn cy(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        for &(q1, q2) in pairs {
            // Direct Pauli action, including the target-Y phase for XI.
            let affected = self.stabs.col_x[q1].clone();
            let mut both_x = BitSet::new();
            for g in &self.stabs.col_x[q1] {
                if self.stabs.col_x[q2].contains(g) {
                    both_x.insert(g);
                }
            }
            self.stabs.signs_minus ^= &both_x;
            mul_i_for(
                &mut self.stabs.signs_minus,
                &mut self.stabs.signs_i,
                &affected,
            );

            for gens in [&mut self.stabs, &mut self.destabs] {
                let x1 = gens.col_x[q1].clone();
                let x2 = gens.col_x[q2].clone();
                let z2 = gens.col_z[q2].clone();
                toggle_col_x(gens, q2, &x1);
                toggle_col_z(gens, q2, &x1);

                let mut z1_effect = x2;
                z1_effect ^= &z2;
                toggle_col_z(gens, q1, &z1_effect);
            }
        }
        self
    }

    /// Controlled-Z gate. XI -> XZ, IX -> ZX, ZI -> ZI, IZ -> IZ.
    #[inline]
    pub fn cz(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        for &(q1, q2) in pairs {
            xor_intersection_into(
                &self.stabs.col_x[q1],
                &self.stabs.col_x[q2],
                &mut self.stabs.signs_minus,
            );

            for gens in [&mut self.stabs, &mut self.destabs] {
                let x1 = gens.col_x[q1].clone();
                let x2 = gens.col_x[q2].clone();
                toggle_col_z(gens, q2, &x1);
                toggle_col_z(gens, q1, &x2);
            }
        }
        self
    }

    /// Square root of XX gate. XI -> XI, IX -> IX, ZI -> -YX, IZ -> -XY.
    #[inline]
    pub fn sxx(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let mut affected = self.stabs.col_z[q1].clone();
            affected ^= &self.stabs.col_z[q2];
            mul_minus_i_for(
                &mut self.stabs.signs_minus,
                &mut self.stabs.signs_i,
                &affected,
            );

            for gens in [&mut self.stabs, &mut self.destabs] {
                let mut affected = gens.col_z[q1].clone();
                affected ^= &gens.col_z[q2];
                toggle_col_x(gens, q1, &affected);
                toggle_col_x(gens, q2, &affected);
            }
        }
        self
    }

    /// Adjoint square root of XX gate. XI -> XI, IX -> IX, ZI -> YX, IZ -> XY.
    #[inline]
    pub fn sxxdg(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let mut affected = self.stabs.col_z[q1].clone();
            affected ^= &self.stabs.col_z[q2];
            mul_i_for(
                &mut self.stabs.signs_minus,
                &mut self.stabs.signs_i,
                &affected,
            );

            for gens in [&mut self.stabs, &mut self.destabs] {
                let mut affected = gens.col_z[q1].clone();
                affected ^= &gens.col_z[q2];
                toggle_col_x(gens, q1, &affected);
                toggle_col_x(gens, q2, &affected);
            }
        }
        self
    }

    /// Square root of ZZ gate. XI -> YZ, IX -> ZY, ZI -> ZI, IZ -> IZ.
    #[inline]
    pub fn szz(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let mut affected = self.stabs.col_x[q1].clone();
            affected ^= &self.stabs.col_x[q2];
            mul_i_for(
                &mut self.stabs.signs_minus,
                &mut self.stabs.signs_i,
                &affected,
            );

            for gens in [&mut self.stabs, &mut self.destabs] {
                let mut affected = gens.col_x[q1].clone();
                affected ^= &gens.col_x[q2];
                toggle_col_z(gens, q1, &affected);
                toggle_col_z(gens, q2, &affected);
            }
        }
        self
    }

    /// Adjoint square root of ZZ gate. XI -> -YZ, IX -> -ZY, ZI -> ZI, IZ -> IZ.
    #[inline]
    pub fn szzdg(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        for &(q1, q2) in pairs {
            let mut affected = self.stabs.col_x[q1].clone();
            affected ^= &self.stabs.col_x[q2];
            mul_minus_i_for(
                &mut self.stabs.signs_minus,
                &mut self.stabs.signs_i,
                &affected,
            );

            for gens in [&mut self.stabs, &mut self.destabs] {
                let mut affected = gens.col_x[q1].clone();
                affected ^= &gens.col_x[q2];
                toggle_col_z(gens, q1, &affected);
                toggle_col_z(gens, q2, &affected);
            }
        }
        self
    }

    /// Square root of YY gate.
    ///
    /// XI -> -ZY, IX -> -YZ, ZI -> XY, IZ -> YX.
    #[inline]
    pub fn syy(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        for &(q1, q2) in pairs {
            self.apply_syy_signs(q1, q2, false);
            self.apply_syy_bits(q1, q2);
        }
        self
    }

    /// Adjoint square root of YY gate.
    ///
    /// XI -> ZY, IX -> YZ, ZI -> -XY, IZ -> -YX.
    #[inline]
    pub fn syydg(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        for &(q1, q2) in pairs {
            self.apply_syy_signs(q1, q2, true);
            self.apply_syy_bits(q1, q2);
        }
        self
    }

    /// SWAP gate. XI -> IX, IX -> XI, ZI -> IZ, IZ -> ZI.
    #[inline]
    pub fn swap(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        for &(q1, q2) in pairs {
            for gens in [&mut self.stabs, &mut self.destabs] {
                let mut affected_x = gens.col_x[q1].clone();
                affected_x ^= &gens.col_x[q2];
                toggle_col_x(gens, q1, &affected_x);
                toggle_col_x(gens, q2, &affected_x);

                let mut affected_z = gens.col_z[q1].clone();
                affected_z ^= &gens.col_z[q2];
                toggle_col_z(gens, q1, &affected_z);
                toggle_col_z(gens, q2, &affected_z);
            }
        }
        self
    }

    fn apply_syy_bits(&mut self, q1: usize, q2: usize) {
        for gens in [&mut self.stabs, &mut self.destabs] {
            let mut affected = gens.col_x[q1].clone();
            affected ^= &gens.col_z[q1];
            affected ^= &gens.col_x[q2];
            affected ^= &gens.col_z[q2];
            toggle_col_x(gens, q1, &affected);
            toggle_col_x(gens, q2, &affected);
            toggle_col_z(gens, q1, &affected);
            toggle_col_z(gens, q2, &affected);
        }
    }

    fn apply_syy_signs(&mut self, q1: usize, q2: usize, adjoint: bool) {
        let col_x = &self.stabs.col_x;
        let col_z = &self.stabs.col_z;
        let signs_minus = &mut self.stabs.signs_minus;
        let signs_i = &mut self.stabs.signs_i;

        macro_rules! apply_syy_sign {
            ($g:expr, $x1:expr, $z1:expr, $x2:expr, $z2:expr) => {
                if ($x1 != $z1) != ($x2 != $z2) {
                    let use_plus_i = ($z1 != $z2) != adjoint;
                    if use_plus_i {
                        mul_i_for(signs_minus, signs_i, &BitSet::single($g));
                    } else {
                        mul_minus_i_for(signs_minus, signs_i, &BitSet::single($g));
                    }
                }
            };
        }

        for g in &col_x[q1] {
            let x1 = true;
            let z1 = col_z[q1].contains(g);
            let x2 = col_x[q2].contains(g);
            let z2 = col_z[q2].contains(g);
            apply_syy_sign!(g, x1, z1, x2, z2);
        }
        for g in &col_z[q1] {
            if col_x[q1].contains(g) {
                continue;
            }
            let x1 = false;
            let z1 = true;
            let x2 = col_x[q2].contains(g);
            let z2 = col_z[q2].contains(g);
            apply_syy_sign!(g, x1, z1, x2, z2);
        }
        for g in &col_x[q2] {
            if col_x[q1].contains(g) || col_z[q1].contains(g) {
                continue;
            }
            let x2 = true;
            let z2 = col_z[q2].contains(g);
            apply_syy_sign!(g, false, false, x2, z2);
        }
        for g in &col_z[q2] {
            if col_x[q1].contains(g) || col_z[q1].contains(g) || col_x[q2].contains(g) {
                continue;
            }
            apply_syy_sign!(g, false, false, false, true);
        }
    }

    // ==================== Measurement ====================

    /// Measure qubits in the Z basis.
    #[inline]
    pub fn mz(&mut self, qubits: &[usize]) -> Vec<SymbolicMeasurementResult> {
        qubits
            .iter()
            .map(|&q| {
                let result = if self.stabs.col_x[q].is_empty() {
                    self.deterministic_meas(q)
                } else {
                    self.nondeterministic_meas(q)
                };

                self.measurement_history.push(result.clone());
                result
            })
            .collect()
    }

    /// Prepare qubit in |0⟩ (reset). Does not record a measurement.
    ///
    /// See `SymbolicSparseStabVecSet::pz` for detailed documentation.
    pub fn pz(&mut self, q: usize) -> &mut Self {
        if self.stabs.col_x[q].is_empty() {
            self.h(&[q]);
        }
        self.pz_nondeterministic(q);
        self
    }

    /// Project qubit onto +Z eigenstate without recording a measurement.
    ///
    /// Same Gaussian elimination as `nondeterministic_meas` but with
    /// empty sign and no measurement counter increment.
    fn pz_nondeterministic(&mut self, q: usize) {
        let mut anticom_stabs_col = self.stabs.col_x[q].clone();
        let mut anticom_destabs_col = self.destabs.col_x[q].clone();

        let mut smallest_wt = 2 * self.num_qubits + 2;
        let mut removed_id: Option<usize> = None;

        for stab_id in &anticom_stabs_col {
            let weight = self.stabs.row_x[stab_id].len() + self.stabs.row_z[stab_id].len();
            if weight < smallest_wt {
                smallest_wt = weight;
                removed_id = Some(stab_id);
            }
        }

        let id = removed_id.expect("col_x[q] was non-empty");
        anticom_stabs_col.remove(id);
        let removed_row_x = self.stabs.row_x[id].clone();
        let removed_row_z = self.stabs.row_z[id].clone();

        if self.stabs.signs_minus.contains(id) {
            self.stabs.signs_minus ^= &anticom_stabs_col;
        }
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

        let removed_sign = self.stabs.signs[id].clone();
        for g in &anticom_stabs_col {
            self.stabs.signs[g].multiply_assign(&removed_sign);
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
            self.stabs.row_x[g] ^= &removed_row_x;
            self.stabs.row_z[g] ^= &removed_row_z;
        }

        for i in &removed_row_x {
            self.stabs.col_x[i] ^= &anticom_stabs_col;
        }
        for i in &removed_row_z {
            self.stabs.col_z[i] ^= &anticom_stabs_col;
        }

        for i in &self.stabs.row_x[id] {
            self.stabs.col_x[i].remove(id);
        }
        for i in &self.stabs.row_z[id] {
            self.stabs.col_z[i].remove(id);
        }

        self.stabs.col_z[q].insert(id);
        self.stabs.row_x[id].clear();
        self.stabs.row_z[id].clear();
        self.stabs.row_z[id].insert(q);
        self.stabs.signs[id] = SymbolicSign::empty();
        self.stabs.signs_minus.remove(id);
        self.stabs.signs_i.remove(id);

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
    use crate::clifford_matrix_oracle::{
        CliffordMatrixGate, SignedPauli, all_pauli_strings, conjugate_pauli,
    };

    fn assert_same_gens(left: &SymbolicGensBitSet, right: &SymbolicGensBitSet) {
        assert_eq!(left.col_x, right.col_x);
        assert_eq!(left.col_z, right.col_z);
        assert_eq!(left.row_x, right.row_x);
        assert_eq!(left.row_z, right.row_z);
        assert_eq!(left.signs, right.signs);
        assert_eq!(left.signs_minus, right.signs_minus);
        assert_eq!(left.signs_i, right.signs_i);
    }

    fn assert_same_state(left: &SymbolicSparseStab, right: &SymbolicSparseStab) {
        assert_same_gens(&left.stabs, &right.stabs);
        assert_same_gens(&left.destabs, &right.destabs);
        assert_eq!(left.measurement_counter, right.measurement_counter);
        assert_eq!(
            left.measurement_history.as_slice(),
            right.measurement_history.as_slice()
        );
    }

    fn nontrivial_state() -> SymbolicSparseStab {
        let mut sim = SymbolicSparseStab::new(3);
        sim.h(&[0, 1]);
        sim.cx(&[(0, 2), (1, 2)]);
        sim.sz(&[0, 2]);
        sim.h(&[2]);
        sim
    }

    fn check_direct_gate(
        apply_direct: impl FnOnce(&mut SymbolicSparseStab),
        apply_reference: impl FnOnce(&mut SymbolicSparseStab),
    ) {
        let mut direct = nontrivial_state();
        let mut reference = direct.clone();
        apply_direct(&mut direct);
        apply_reference(&mut reference);
        assert_same_state(&direct, &reference);
    }

    fn row_binary(gens: &SymbolicGensBitSet, row: usize, num_qubits: usize) -> String {
        let mut dense = String::with_capacity(num_qubits);
        for q in 0..num_qubits {
            dense.push(
                match (gens.row_x[row].contains(q), gens.row_z[row].contains(q)) {
                    (false, false) => 'I',
                    (true, false) => 'X',
                    (false, true) => 'Z',
                    (true, true) => 'Y',
                },
            );
        }
        dense
    }

    fn assert_symbolic_single_qubit_gate_basis(
        apply: impl FnOnce(&mut SymbolicSparseStab),
        expected_x: &str,
        expected_z: &str,
    ) {
        let mut sim = SymbolicSparseStab::new(1);
        apply(&mut sim);

        assert_eq!(row_binary(&sim.destabs, 0, 1), expected_x, "X image");
        assert_eq!(row_binary(&sim.stabs, 0, 1), expected_z, "Z image");
    }

    fn assert_symbolic_two_qubit_gate_basis(
        apply: impl FnOnce(&mut SymbolicSparseStab),
        expected_xi: &str,
        expected_ix: &str,
        expected_zi: &str,
        expected_iz: &str,
    ) {
        let mut sim = SymbolicSparseStab::new(2);
        apply(&mut sim);

        assert_eq!(row_binary(&sim.destabs, 0, 2), expected_xi, "XI image");
        assert_eq!(row_binary(&sim.destabs, 1, 2), expected_ix, "IX image");
        assert_eq!(row_binary(&sim.stabs, 0, 2), expected_zi, "ZI image");
        assert_eq!(row_binary(&sim.stabs, 1, 2), expected_iz, "IZ image");
    }

    fn set_stab_row_to_pauli(gens: &mut SymbolicGensBitSet, row: usize, pauli: &str) {
        let mut y_count = 0usize;
        for (q, label) in pauli.chars().enumerate() {
            match label {
                'I' => {}
                'X' => {
                    gens.row_x[row].insert(q);
                    gens.col_x[q].insert(row);
                }
                'Y' => {
                    y_count += 1;
                    gens.row_x[row].insert(q);
                    gens.row_z[row].insert(q);
                    gens.col_x[q].insert(row);
                    gens.col_z[q].insert(row);
                }
                'Z' => {
                    gens.row_z[row].insert(q);
                    gens.col_z[q].insert(row);
                }
                _ => panic!("invalid Pauli label {label}"),
            }
        }

        match y_count % 4 {
            0 => {}
            1 => {
                gens.signs_i.insert(row);
            }
            2 => {
                gens.signs_minus.insert(row);
            }
            3 => {
                gens.signs_minus.insert(row);
                gens.signs_i.insert(row);
            }
            _ => unreachable!(),
        }
    }

    fn signed_stab_row(gens: &SymbolicGensBitSet, row: usize, num_qubits: usize) -> SignedPauli {
        assert!(
            gens.signs[row].measurements.is_empty(),
            "unexpected measurement-dependent sign"
        );
        let pauli = row_binary(gens, row, num_qubits);
        let y_count = pauli.chars().filter(|&label| label == 'Y').count();
        let internal_phase = usize::from(gens.signs_i.contains(row))
            + if gens.signs_minus.contains(row) { 2 } else { 0 };
        let canonical_phase = (internal_phase + 3 * y_count) % 4;
        assert!(
            canonical_phase == 0 || canonical_phase == 2,
            "unexpected non-Hermitian phase i^{canonical_phase} for row {row}"
        );
        SignedPauli {
            sign: if canonical_phase == 2 { -1 } else { 1 },
            pauli,
        }
    }

    fn symbolic_image_for_pauli<F>(num_qubits: usize, input: &str, apply: F) -> SignedPauli
    where
        F: FnOnce(&mut SymbolicSparseStab),
    {
        let mut sim = SymbolicSparseStab::new(num_qubits);
        sim.stabs = SymbolicGensBitSet::new(num_qubits);
        sim.destabs = SymbolicGensBitSet::new(num_qubits);
        set_stab_row_to_pauli(&mut sim.stabs, 0, input);
        apply(&mut sim);
        signed_stab_row(&sim.stabs, 0, num_qubits)
    }

    fn assert_symbolic_gate_matches_matrix_oracle<F>(
        name: &str,
        gate: CliffordMatrixGate,
        num_qubits: usize,
        apply: F,
    ) where
        F: Fn(&mut SymbolicSparseStab) + Copy,
    {
        for input in all_pauli_strings(num_qubits) {
            let expected = conjugate_pauli(gate, &input);
            let observed = symbolic_image_for_pauli(num_qubits, &input, apply);
            assert_eq!(observed, expected, "{name}: {input}");
        }
    }

    fn reverse_two_qubit_pauli(pauli: &str) -> String {
        let labels: Vec<char> = pauli.chars().collect();
        assert_eq!(labels.len(), 2);
        [labels[1], labels[0]].into_iter().collect()
    }

    fn assert_symbolic_reversed_pair_matches_matrix_oracle<F>(
        name: &str,
        gate: CliffordMatrixGate,
        apply: F,
    ) where
        F: Fn(&mut SymbolicSparseStab, &[(usize, usize)]) + Copy,
    {
        for input in all_pauli_strings(2) {
            let oracle_input = reverse_two_qubit_pauli(&input);
            let mut expected = conjugate_pauli(gate, &oracle_input);
            expected.pauli = reverse_two_qubit_pauli(&expected.pauli);

            let observed = symbolic_image_for_pauli(2, &input, |sim| {
                apply(sim, &[(1, 0)]);
            });
            assert_eq!(observed, expected, "{name} reversed pair: {input}");
        }
    }

    fn assert_symbolic_two_pair_batch_matches_sequential<F>(name: &str, apply: F)
    where
        F: Fn(&mut SymbolicSparseStab, &[(usize, usize)]) + Copy,
    {
        for input in all_pauli_strings(4) {
            let batched = symbolic_image_for_pauli(4, &input, |sim| {
                apply(sim, &[(0, 1), (2, 3)]);
            });
            let sequential = symbolic_image_for_pauli(4, &input, |sim| {
                apply(sim, &[(0, 1)]);
                apply(sim, &[(2, 3)]);
            });
            assert_eq!(batched, sequential, "{name} batched: {input}");
        }
    }

    fn ref_szdg(sim: &mut SymbolicSparseStab, qs: &[usize]) {
        sim.z(qs);
        sim.sz(qs);
    }

    fn ref_sx(sim: &mut SymbolicSparseStab, qs: &[usize]) {
        sim.h(qs);
        sim.sz(qs);
        sim.h(qs);
    }

    fn ref_sxdg(sim: &mut SymbolicSparseStab, qs: &[usize]) {
        sim.h(qs);
        ref_szdg(sim, qs);
        sim.h(qs);
    }

    fn ref_sy(sim: &mut SymbolicSparseStab, qs: &[usize]) {
        sim.z(qs);
        sim.h(qs);
    }

    fn ref_sydg(sim: &mut SymbolicSparseStab, qs: &[usize]) {
        sim.h(qs);
        sim.z(qs);
    }

    fn ref_cy(sim: &mut SymbolicSparseStab, pairs: &[(usize, usize)]) {
        let targets: Vec<usize> = pairs.iter().map(|&(_, q2)| q2).collect();
        ref_szdg(sim, &targets);
        sim.cx(pairs);
        sim.sz(&targets);
    }

    fn ref_cz(sim: &mut SymbolicSparseStab, pairs: &[(usize, usize)]) {
        let targets: Vec<usize> = pairs.iter().map(|&(_, q2)| q2).collect();
        sim.h(&targets);
        sim.cx(pairs);
        sim.h(&targets);
    }

    fn ref_sxx(sim: &mut SymbolicSparseStab, pairs: &[(usize, usize)]) {
        let q1s: Vec<usize> = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: Vec<usize> = pairs.iter().map(|&(_, q2)| q2).collect();
        ref_sx(sim, &q1s);
        ref_sx(sim, &q2s);
        ref_sydg(sim, &q1s);
        sim.cx(pairs);
        ref_sy(sim, &q1s);
    }

    fn ref_sxxdg(sim: &mut SymbolicSparseStab, pairs: &[(usize, usize)]) {
        let q1s: Vec<usize> = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: Vec<usize> = pairs.iter().map(|&(_, q2)| q2).collect();
        sim.x(&q1s);
        sim.x(&q2s);
        ref_sxx(sim, pairs);
    }

    fn ref_syy(sim: &mut SymbolicSparseStab, pairs: &[(usize, usize)]) {
        let q1s: Vec<usize> = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: Vec<usize> = pairs.iter().map(|&(_, q2)| q2).collect();
        ref_szdg(sim, &q1s);
        ref_szdg(sim, &q2s);
        ref_sxx(sim, pairs);
        sim.sz(&q1s);
        sim.sz(&q2s);
    }

    fn ref_syydg(sim: &mut SymbolicSparseStab, pairs: &[(usize, usize)]) {
        let q1s: Vec<usize> = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: Vec<usize> = pairs.iter().map(|&(_, q2)| q2).collect();
        sim.y(&q1s);
        sim.y(&q2s);
        ref_syy(sim, pairs);
    }

    fn ref_szz(sim: &mut SymbolicSparseStab, pairs: &[(usize, usize)]) {
        let q1s: Vec<usize> = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: Vec<usize> = pairs.iter().map(|&(_, q2)| q2).collect();
        sim.h(&q1s);
        sim.h(&q2s);
        ref_sxx(sim, pairs);
        sim.h(&q1s);
        sim.h(&q2s);
    }

    fn ref_szzdg(sim: &mut SymbolicSparseStab, pairs: &[(usize, usize)]) {
        let q1s: Vec<usize> = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: Vec<usize> = pairs.iter().map(|&(_, q2)| q2).collect();
        sim.z(&q1s);
        sim.z(&q2s);
        ref_szz(sim, pairs);
    }

    fn ref_swap(sim: &mut SymbolicSparseStab, pairs: &[(usize, usize)]) {
        let reversed: Vec<(usize, usize)> = pairs.iter().map(|&(q1, q2)| (q2, q1)).collect();
        sim.cx(pairs);
        sim.cx(&reversed);
        sim.cx(pairs);
    }

    #[test]
    fn test_bell_state_bitset() {
        let mut sim = SymbolicSparseStab::new(2);

        // Create Bell state
        sim.h(&[0]).cx(&[(0, 1)]);

        // Measure qubit 0 - should be non-deterministic
        let r0 = sim.mz(&[0])[0].clone();
        assert!(!r0.is_deterministic);
        assert_eq!(r0.outcome.len(), 1);
        assert!(r0.outcome.contains(0));
        assert_eq!(r0.index, 0);

        // Measure qubit 1 - should be deterministic
        let r1 = sim.mz(&[1])[0].clone();
        assert!(r1.is_deterministic);
        assert_eq!(r0.outcome, r1.outcome);
        assert_eq!(r1.index, 1);
    }

    #[test]
    fn test_product_state_bitset() {
        let mut sim = SymbolicSparseStab::new(2);

        let r0 = sim.mz(&[0])[0].clone();
        assert!(r0.is_deterministic);
        assert!(r0.outcome.is_empty());
        assert_eq!(r0.index, 0);

        let r1 = sim.mz(&[1])[0].clone();
        assert!(r1.is_deterministic);
        assert!(r1.outcome.is_empty());
        assert_eq!(r1.index, 1);
    }

    #[test]
    fn test_direct_clifford_gate_binary_truth_tables() {
        assert_symbolic_single_qubit_gate_basis(
            |sim| {
                sim.szdg(&[0]);
            },
            "Y",
            "Z",
        );
        assert_symbolic_single_qubit_gate_basis(
            |sim| {
                sim.sx(&[0]);
            },
            "X",
            "Y",
        );
        assert_symbolic_single_qubit_gate_basis(
            |sim| {
                sim.sxdg(&[0]);
            },
            "X",
            "Y",
        );
        assert_symbolic_single_qubit_gate_basis(
            |sim| {
                sim.sy(&[0]);
            },
            "Z",
            "X",
        );
        assert_symbolic_single_qubit_gate_basis(
            |sim| {
                sim.sydg(&[0]);
            },
            "Z",
            "X",
        );
        assert_symbolic_two_qubit_gate_basis(
            |sim| {
                sim.cy(&[(0, 1)]);
            },
            "XY",
            "ZX",
            "ZI",
            "ZZ",
        );
        assert_symbolic_two_qubit_gate_basis(
            |sim| {
                sim.cz(&[(0, 1)]);
            },
            "XZ",
            "ZX",
            "ZI",
            "IZ",
        );
        assert_symbolic_two_qubit_gate_basis(
            |sim| {
                sim.sxx(&[(0, 1)]);
            },
            "XI",
            "IX",
            "YX",
            "XY",
        );
        assert_symbolic_two_qubit_gate_basis(
            |sim| {
                sim.sxxdg(&[(0, 1)]);
            },
            "XI",
            "IX",
            "YX",
            "XY",
        );
        assert_symbolic_two_qubit_gate_basis(
            |sim| {
                sim.syy(&[(0, 1)]);
            },
            "ZY",
            "YZ",
            "XY",
            "YX",
        );
        assert_symbolic_two_qubit_gate_basis(
            |sim| {
                sim.syydg(&[(0, 1)]);
            },
            "ZY",
            "YZ",
            "XY",
            "YX",
        );
        assert_symbolic_two_qubit_gate_basis(
            |sim| {
                sim.szz(&[(0, 1)]);
            },
            "YZ",
            "ZY",
            "ZI",
            "IZ",
        );
        assert_symbolic_two_qubit_gate_basis(
            |sim| {
                sim.szzdg(&[(0, 1)]);
            },
            "YZ",
            "ZY",
            "ZI",
            "IZ",
        );
        assert_symbolic_two_qubit_gate_basis(
            |sim| {
                sim.swap(&[(0, 1)]);
            },
            "IX",
            "XI",
            "IZ",
            "ZI",
        );
    }

    #[test]
    fn test_direct_clifford_gates_match_matrix_oracle_for_all_paulis() {
        assert_symbolic_gate_matches_matrix_oracle("CX", CliffordMatrixGate::CX, 2, |sim| {
            sim.cx(&[(0, 1)]);
        });
        assert_symbolic_gate_matches_matrix_oracle("SZdg", CliffordMatrixGate::SZdg, 1, |sim| {
            sim.szdg(&[0]);
        });
        assert_symbolic_gate_matches_matrix_oracle("F", CliffordMatrixGate::F, 1, |sim| {
            sim.sx(&[0]);
            sim.sz(&[0]);
        });
        assert_symbolic_gate_matches_matrix_oracle("Fdg", CliffordMatrixGate::Fdg, 1, |sim| {
            sim.szdg(&[0]);
            sim.sxdg(&[0]);
        });
        assert_symbolic_gate_matches_matrix_oracle("SX", CliffordMatrixGate::SX, 1, |sim| {
            sim.sx(&[0]);
        });
        assert_symbolic_gate_matches_matrix_oracle("SXdg", CliffordMatrixGate::SXdg, 1, |sim| {
            sim.sxdg(&[0]);
        });
        assert_symbolic_gate_matches_matrix_oracle("SY", CliffordMatrixGate::SY, 1, |sim| {
            sim.sy(&[0]);
        });
        assert_symbolic_gate_matches_matrix_oracle("SYdg", CliffordMatrixGate::SYdg, 1, |sim| {
            sim.sydg(&[0]);
        });
        assert_symbolic_gate_matches_matrix_oracle("CY", CliffordMatrixGate::CY, 2, |sim| {
            sim.cy(&[(0, 1)]);
        });
        assert_symbolic_gate_matches_matrix_oracle("CZ", CliffordMatrixGate::CZ, 2, |sim| {
            sim.cz(&[(0, 1)]);
        });
        assert_symbolic_gate_matches_matrix_oracle("SXX", CliffordMatrixGate::SXX, 2, |sim| {
            sim.sxx(&[(0, 1)]);
        });
        assert_symbolic_gate_matches_matrix_oracle("SXXdg", CliffordMatrixGate::SXXdg, 2, |sim| {
            sim.sxxdg(&[(0, 1)]);
        });
        assert_symbolic_gate_matches_matrix_oracle("SYY", CliffordMatrixGate::SYY, 2, |sim| {
            sim.syy(&[(0, 1)]);
        });
        assert_symbolic_gate_matches_matrix_oracle("SYYdg", CliffordMatrixGate::SYYdg, 2, |sim| {
            sim.syydg(&[(0, 1)]);
        });
        assert_symbolic_gate_matches_matrix_oracle("SZZ", CliffordMatrixGate::SZZ, 2, |sim| {
            sim.szz(&[(0, 1)]);
        });
        assert_symbolic_gate_matches_matrix_oracle("SZZdg", CliffordMatrixGate::SZZdg, 2, |sim| {
            sim.szzdg(&[(0, 1)]);
        });
        assert_symbolic_gate_matches_matrix_oracle("SWAP", CliffordMatrixGate::SWAP, 2, |sim| {
            sim.swap(&[(0, 1)]);
        });
    }

    #[test]
    fn test_cy_xx_sign_regression() {
        assert_eq!(
            symbolic_image_for_pauli(2, "XX", |sim| {
                sim.cy(&[(0, 1)]);
            }),
            SignedPauli {
                sign: -1,
                pauli: "YZ".to_string()
            }
        );
    }

    #[test]
    fn test_two_qubit_gates_reversed_pair_matches_matrix_oracle() {
        assert_symbolic_reversed_pair_matches_matrix_oracle(
            "CX",
            CliffordMatrixGate::CX,
            |sim, pairs| {
                sim.cx(pairs);
            },
        );
        assert_symbolic_reversed_pair_matches_matrix_oracle(
            "CY",
            CliffordMatrixGate::CY,
            |sim, pairs| {
                sim.cy(pairs);
            },
        );
        assert_symbolic_reversed_pair_matches_matrix_oracle(
            "CZ",
            CliffordMatrixGate::CZ,
            |sim, pairs| {
                sim.cz(pairs);
            },
        );
        assert_symbolic_reversed_pair_matches_matrix_oracle(
            "SXX",
            CliffordMatrixGate::SXX,
            |sim, pairs| {
                sim.sxx(pairs);
            },
        );
        assert_symbolic_reversed_pair_matches_matrix_oracle(
            "SXXdg",
            CliffordMatrixGate::SXXdg,
            |sim, pairs| {
                sim.sxxdg(pairs);
            },
        );
        assert_symbolic_reversed_pair_matches_matrix_oracle(
            "SYY",
            CliffordMatrixGate::SYY,
            |sim, pairs| {
                sim.syy(pairs);
            },
        );
        assert_symbolic_reversed_pair_matches_matrix_oracle(
            "SYYdg",
            CliffordMatrixGate::SYYdg,
            |sim, pairs| {
                sim.syydg(pairs);
            },
        );
        assert_symbolic_reversed_pair_matches_matrix_oracle(
            "SZZ",
            CliffordMatrixGate::SZZ,
            |sim, pairs| {
                sim.szz(pairs);
            },
        );
        assert_symbolic_reversed_pair_matches_matrix_oracle(
            "SZZdg",
            CliffordMatrixGate::SZZdg,
            |sim, pairs| {
                sim.szzdg(pairs);
            },
        );
        assert_symbolic_reversed_pair_matches_matrix_oracle(
            "SWAP",
            CliffordMatrixGate::SWAP,
            |sim, pairs| {
                sim.swap(pairs);
            },
        );
    }

    #[test]
    fn test_two_qubit_gate_batches_match_sequential_pairs() {
        assert_symbolic_two_pair_batch_matches_sequential("CX", |sim, pairs| {
            sim.cx(pairs);
        });
        assert_symbolic_two_pair_batch_matches_sequential("CY", |sim, pairs| {
            sim.cy(pairs);
        });
        assert_symbolic_two_pair_batch_matches_sequential("CZ", |sim, pairs| {
            sim.cz(pairs);
        });
        assert_symbolic_two_pair_batch_matches_sequential("SXX", |sim, pairs| {
            sim.sxx(pairs);
        });
        assert_symbolic_two_pair_batch_matches_sequential("SXXdg", |sim, pairs| {
            sim.sxxdg(pairs);
        });
        assert_symbolic_two_pair_batch_matches_sequential("SYY", |sim, pairs| {
            sim.syy(pairs);
        });
        assert_symbolic_two_pair_batch_matches_sequential("SYYdg", |sim, pairs| {
            sim.syydg(pairs);
        });
        assert_symbolic_two_pair_batch_matches_sequential("SZZ", |sim, pairs| {
            sim.szz(pairs);
        });
        assert_symbolic_two_pair_batch_matches_sequential("SZZdg", |sim, pairs| {
            sim.szzdg(pairs);
        });
        assert_symbolic_two_pair_batch_matches_sequential("SWAP", |sim, pairs| {
            sim.swap(pairs);
        });
    }

    #[test]
    fn test_direct_clifford_gates_match_reference_sequences() {
        check_direct_gate(
            |sim| {
                sim.szdg(&[0]);
            },
            |sim| ref_szdg(sim, &[0]),
        );
        check_direct_gate(
            |sim| {
                sim.sx(&[0]);
            },
            |sim| ref_sx(sim, &[0]),
        );
        check_direct_gate(
            |sim| {
                sim.sxdg(&[0]);
            },
            |sim| ref_sxdg(sim, &[0]),
        );
        check_direct_gate(
            |sim| {
                sim.sy(&[0]);
            },
            |sim| ref_sy(sim, &[0]),
        );
        check_direct_gate(
            |sim| {
                sim.sydg(&[0]);
            },
            |sim| ref_sydg(sim, &[0]),
        );
        check_direct_gate(
            |sim| {
                sim.cy(&[(0, 1)]);
            },
            |sim| ref_cy(sim, &[(0, 1)]),
        );
        check_direct_gate(
            |sim| {
                sim.cz(&[(0, 1)]);
            },
            |sim| ref_cz(sim, &[(0, 1)]),
        );
        check_direct_gate(
            |sim| {
                sim.sxx(&[(0, 1)]);
            },
            |sim| ref_sxx(sim, &[(0, 1)]),
        );
        check_direct_gate(
            |sim| {
                sim.sxxdg(&[(0, 1)]);
            },
            |sim| ref_sxxdg(sim, &[(0, 1)]),
        );
        check_direct_gate(
            |sim| {
                sim.syy(&[(0, 1)]);
            },
            |sim| ref_syy(sim, &[(0, 1)]),
        );
        check_direct_gate(
            |sim| {
                sim.syydg(&[(0, 1)]);
            },
            |sim| ref_syydg(sim, &[(0, 1)]),
        );
        check_direct_gate(
            |sim| {
                sim.szz(&[(0, 1)]);
            },
            |sim| ref_szz(sim, &[(0, 1)]),
        );
        check_direct_gate(
            |sim| {
                sim.szzdg(&[(0, 1)]);
            },
            |sim| ref_szzdg(sim, &[(0, 1)]),
        );
        check_direct_gate(
            |sim| {
                sim.swap(&[(0, 1)]);
            },
            |sim| ref_swap(sim, &[(0, 1)]),
        );
    }
}
