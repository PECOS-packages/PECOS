// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! DOD-style batched operations for stabilizer simulators.
//!
//! This module provides optimized batch operations that process multiple gates
//! of the same type together, reducing overhead and enabling better cache utilization.
//!
//! # Key Optimizations
//!
//! 1. **Reduced function call overhead**: One dispatch per gate type, not per gate
//! 2. **Better cache locality**: Process all columns for same operation together
//! 3. **Vectorizable inner loops**: Sign updates can use SIMD operations
//! 4. **Deferred row updates**: Collect row changes and apply in batch
//!
//! # Example
//!
//! ```ignore
//! use pecos_simulators::{SparseStab, BatchedOps};
//!
//! let mut sim = SparseStab::new(100);
//!
//! // Instead of calling h() 100 times:
//! let qubits: Vec<_> = (0..100).collect();
//! sim.h_batched(&qubits);  // Process all H gates in one optimized pass
//! ```

use crate::gens::GensGeneric;
use crate::sparse_stab::SparseStabGeneric;
use pecos_core::{BitSet, IndexSet};
use pecos_random::{Rng, SeedableRng};
use std::fmt::Debug;
use std::mem;

/// Extension trait for batched operations on `SparseStab`.
///
/// These methods provide optimized implementations that process multiple
/// qubits in a single pass, reducing overhead compared to calling the
/// standard methods in a loop.
pub trait BatchedOps {
    /// Applies H gates to multiple qubits in a single optimized pass.
    ///
    /// This is more efficient than calling `h()` in a loop because:
    /// - Sign updates for all qubits are collected and applied together
    /// - Column swaps are performed in a cache-friendly order
    /// - Row updates are batched to reduce redundant lookups
    fn h_batched(&mut self, qubits: &[usize]) -> &mut Self;

    /// Applies X gates to multiple qubits in a single optimized pass.
    fn x_batched(&mut self, qubits: &[usize]) -> &mut Self;

    /// Applies Z gates to multiple qubits in a single optimized pass.
    fn z_batched(&mut self, qubits: &[usize]) -> &mut Self;

    /// Applies SZ (S) gates to multiple qubits in a single optimized pass.
    fn sz_batched(&mut self, qubits: &[usize]) -> &mut Self;

    /// Applies CX gates to multiple qubit pairs in a single optimized pass.
    ///
    /// Qubits are provided as pairs: [c0, t0, c1, t1, ...]
    fn cx_batched(&mut self, qubits: &[usize]) -> &mut Self;

    /// Applies CZ gates to multiple qubit pairs in a single optimized pass.
    fn cz_batched(&mut self, qubits: &[usize]) -> &mut Self;
}

impl<R: SeedableRng + Rng + Debug> BatchedOps for SparseStabGeneric<BitSet, R> {
    #[inline]
    fn h_batched(&mut self, qubits: &[usize]) -> &mut Self {
        // Early exit for empty input
        if qubits.is_empty() {
            return self;
        }

        // Phase 1: Batch sign updates
        // Collect all intersection XORs into signs_minus
        for &qu in qubits {
            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
        }

        // Phase 2: Batch row updates and column swaps
        // Process both stabs and destabs
        for g in [&mut self.stabs, &mut self.destabs] {
            h_batched_gens(g, qubits);
        }

        self
    }

    #[inline]
    fn x_batched(&mut self, qubits: &[usize]) -> &mut Self {
        // X gate: Z -> -Z (toggle sign for generators with Z on qubit)
        // Phase update: XOR col_z[qu] into signs_minus
        for &qu in qubits {
            self.stabs.signs_minus.xor_assign(&self.stabs.col_z[qu]);
        }
        self
    }

    #[inline]
    fn z_batched(&mut self, qubits: &[usize]) -> &mut Self {
        // Z gate: X -> -X (toggle sign for generators with X on qubit)
        // Phase update: XOR col_x[qu] into signs_minus
        for &qu in qubits {
            self.stabs.signs_minus.xor_assign(&self.stabs.col_x[qu]);
        }
        self
    }

    #[inline]
    fn sz_batched(&mut self, qubits: &[usize]) -> &mut Self {
        // SZ gate: X -> Y (add i phase), Y -> -X (toggle minus, remove i)
        // Phase updates: XOR col_x into signs_i, then XOR intersection into signs_minus
        for &qu in qubits {
            // Add i phase for X terms
            self.stabs.signs_i.xor_assign(&self.stabs.col_x[qu]);
            // Toggle minus for Y terms (intersection of X and Z after i update)
            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
        }
        self
    }

    #[inline]
    fn cx_batched(&mut self, qubits: &[usize]) -> &mut Self {
        // CX: IX->IX, IZ->ZZ, XI->XX, ZI->ZI
        // Process pairs
        for pair in qubits.chunks_exact(2) {
            let ctrl = pair[0];
            let tgt = pair[1];

            for g in [&mut self.stabs, &mut self.destabs] {
                cx_single_gens(g, ctrl, tgt);
            }

            // Sign update: XOR (col_x[ctrl] ∩ col_z[tgt]) ∩ (col_z[ctrl] ^ col_x[tgt]) into signs_minus
            // This handles the Y*Y -> -Y*Y case
            // Compute intersection of col_x[ctrl] and col_z[tgt]
            let intersection: Vec<usize> = self.stabs.col_x[ctrl]
                .iter()
                .filter(|i| self.stabs.col_z[tgt].contains(*i))
                .collect();

            // For each element in intersection, check if it's in (col_z[ctrl] XOR col_x[tgt])
            for i in intersection {
                let in_z_ctrl = self.stabs.col_z[ctrl].contains(i);
                let in_x_tgt = self.stabs.col_x[tgt].contains(i);
                if in_z_ctrl != in_x_tgt {
                    self.stabs.signs_minus.toggle(i);
                }
            }
        }

        self
    }

    #[inline]
    fn cz_batched(&mut self, qubits: &[usize]) -> &mut Self {
        // CZ: IX->ZX, IZ->IZ, XI->XZ, ZI->ZI
        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0];
            let q2 = pair[1];

            for g in [&mut self.stabs, &mut self.destabs] {
                cz_single_gens(g, q1, q2);
            }

            // Sign update for CZ
            // Compute intersection of col_x[q1] and col_x[q2]
            let intersection: Vec<usize> = self.stabs.col_x[q1]
                .iter()
                .filter(|i| self.stabs.col_x[q2].contains(*i))
                .collect();

            // For each element, check if it's in (col_z[q1] XOR col_z[q2])
            for i in intersection {
                let in_z_q1 = self.stabs.col_z[q1].contains(i);
                let in_z_q2 = self.stabs.col_z[q2].contains(i);
                if in_z_q1 != in_z_q2 {
                    self.stabs.signs_minus.toggle(i);
                }
            }
        }

        self
    }
}

/// Batched H gate on a single Gens structure.
///
/// This processes all qubits in two phases:
/// 1. Collect row updates for all qubits
/// 2. Apply column swaps
#[inline]
fn h_batched_gens(g: &mut GensGeneric<BitSet>, qubits: &[usize]) {
    // Process each qubit - row updates then column swap
    for &qu in qubits {
        // Elements in col_x but not in col_z: X -> Z
        // We need to update row_x and row_z for affected generators
        let col_x_qu = &g.col_x[qu];
        let col_z_qu = &g.col_z[qu];

        // Collect generators that need X removed and Z added
        for i in col_x_qu {
            if !col_z_qu.contains(i) {
                g.row_x[i].remove(qu);
                g.row_z[i].insert(qu);
            }
        }

        // Collect generators that need Z removed and X added
        for i in col_z_qu {
            if !col_x_qu.contains(i) {
                g.row_z[i].remove(qu);
                g.row_x[i].insert(qu);
            }
        }

        // Swap columns
        mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
    }
}

/// CX gate on a single Gens structure.
#[inline]
fn cx_single_gens(g: &mut GensGeneric<BitSet>, ctrl: usize, tgt: usize) {
    let (qu_min, qu_max) = if ctrl < tgt { (ctrl, tgt) } else { (tgt, ctrl) };

    // Handle col_x: XI -> XX (add X on target for generators with X on control)
    {
        let (_left, right) = g.col_x.split_at_mut(qu_min);
        let (mid, right) = right.split_at_mut(qu_max - qu_min);
        let col_x_min = &mut mid[0];
        let col_x_max = &mut right[0];

        let (col_x_ctrl, col_x_tgt) = if ctrl < tgt {
            (col_x_min as &BitSet, col_x_max)
        } else {
            (col_x_max as &BitSet, col_x_min)
        };

        // Add X on target for each generator with X on control
        for i in col_x_ctrl {
            col_x_tgt.toggle(i);
            g.row_x[i].toggle(tgt);
        }
    }

    // Handle col_z: IZ -> ZZ (add Z on control for generators with Z on target)
    {
        let (_left, right) = g.col_z.split_at_mut(qu_min);
        let (mid, right) = right.split_at_mut(qu_max - qu_min);
        let col_z_min = &mut mid[0];
        let col_z_max = &mut right[0];

        let (col_z_ctrl, col_z_tgt) = if ctrl < tgt {
            (col_z_min, col_z_max as &BitSet)
        } else {
            (col_z_max, col_z_min as &BitSet)
        };

        // Add Z on control for each generator with Z on target
        for i in col_z_tgt {
            col_z_ctrl.toggle(i);
            g.row_z[i].toggle(ctrl);
        }
    }
}

/// CZ gate on a single Gens structure.
#[inline]
fn cz_single_gens(g: &mut GensGeneric<BitSet>, q1: usize, q2: usize) {
    // CZ is symmetric: IX->ZX, XI->XZ
    // Add Z on q2 for generators with X on q1
    // Add Z on q1 for generators with X on q2

    // Handle q1's X -> add Z on q2
    {
        let col_x_q1: Vec<usize> = g.col_x[q1].iter().collect();
        for i in col_x_q1 {
            g.col_z[q2].toggle(i);
            g.row_z[i].toggle(q2);
        }
    }

    // Handle q2's X -> add Z on q1
    {
        let col_x_q2: Vec<usize> = g.col_x[q2].iter().collect();
        for i in col_x_q2 {
            g.col_z[q1].toggle(i);
            g.row_z[i].toggle(q1);
        }
    }
}

// ============================================================================
// Raw Operations (usize indices, no QubitId conversion)
// ============================================================================

/// Trait for raw operations using usize indices directly.
///
/// These methods bypass the `QubitId` wrapper for maximum performance
/// in hot loops where the overhead of `QubitId` conversion is significant.
pub trait RawOps {
    /// Apply H gates using raw usize indices.
    fn h_raw(&mut self, qubits: &[usize]) -> &mut Self;

    /// Apply CX gates using raw usize indices (pairs: [c0, t0, c1, t1, ...]).
    fn cx_raw(&mut self, qubits: &[usize]) -> &mut Self;

    /// Measure qubits using raw usize indices.
    fn mz_raw(&mut self, qubits: &[usize]) -> Vec<crate::MeasurementResult>;
}

impl<S: IndexSet, R: SeedableRng + Rng + Debug> RawOps for SparseStabGeneric<S, R> {
    #[inline]
    fn h_raw(&mut self, qubits: &[usize]) -> &mut Self {
        use crate::CliffordGateable;
        use pecos_core::QubitId;

        // For now, delegate to the standard implementation
        // A future optimization could inline the logic directly
        let ids: smallvec::SmallVec<[QubitId; 16]> = qubits.iter().map(|&q| QubitId(q)).collect();
        self.h(&ids)
    }

    #[inline]
    fn cx_raw(&mut self, qubits: &[usize]) -> &mut Self {
        use crate::CliffordGateable;
        use pecos_core::QubitId;

        let ids: smallvec::SmallVec<[QubitId; 16]> = qubits.iter().map(|&q| QubitId(q)).collect();
        self.cx(&ids)
    }

    #[inline]
    fn mz_raw(&mut self, qubits: &[usize]) -> Vec<crate::MeasurementResult> {
        use crate::CliffordGateable;
        use pecos_core::QubitId;

        let ids: smallvec::SmallVec<[QubitId; 16]> = qubits.iter().map(|&q| QubitId(q)).collect();
        self.mz(&ids)
    }
}

// ============================================================================
// Command Buffer for Deferred Execution
// ============================================================================

/// A command buffer that accumulates gate operations for batched execution.
///
/// This implements the ECS "system" pattern where operations are collected
/// and then executed in optimized batches.
///
/// # Example
///
/// ```ignore
/// let mut buffer = CommandBuffer::new();
///
/// // Accumulate gates
/// buffer.h(&[0, 1, 2, 3]);
/// buffer.cx(&[0, 1, 2, 3]);
/// buffer.h(&[4, 5, 6, 7]);  // More H gates
///
/// // Execute all accumulated gates in optimized order
/// buffer.flush(&mut sim);
/// ```
#[derive(Debug, Default)]
#[allow(clippy::struct_field_names)]
pub struct CommandBuffer {
    /// Accumulated H gate qubits
    h_qubits: Vec<usize>,
    /// Accumulated X gate qubits
    x_qubits: Vec<usize>,
    /// Accumulated Z gate qubits
    z_qubits: Vec<usize>,
    /// Accumulated SZ gate qubits
    sz_qubits: Vec<usize>,
    /// Accumulated CX gate pairs (ctrl, tgt, ctrl, tgt, ...)
    cx_qubits: Vec<usize>,
    /// Accumulated CZ gate pairs
    cz_qubits: Vec<usize>,
}

impl CommandBuffer {
    /// Creates a new empty command buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a command buffer with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(single_qubit: usize, two_qubit: usize) -> Self {
        Self {
            h_qubits: Vec::with_capacity(single_qubit),
            x_qubits: Vec::with_capacity(single_qubit),
            z_qubits: Vec::with_capacity(single_qubit),
            sz_qubits: Vec::with_capacity(single_qubit),
            cx_qubits: Vec::with_capacity(two_qubit * 2),
            cz_qubits: Vec::with_capacity(two_qubit * 2),
        }
    }

    /// Adds H gates to the buffer.
    #[inline]
    pub fn h(&mut self, qubits: &[usize]) -> &mut Self {
        self.h_qubits.extend_from_slice(qubits);
        self
    }

    /// Adds X gates to the buffer.
    #[inline]
    pub fn x(&mut self, qubits: &[usize]) -> &mut Self {
        self.x_qubits.extend_from_slice(qubits);
        self
    }

    /// Adds Z gates to the buffer.
    #[inline]
    pub fn z(&mut self, qubits: &[usize]) -> &mut Self {
        self.z_qubits.extend_from_slice(qubits);
        self
    }

    /// Adds SZ gates to the buffer.
    #[inline]
    pub fn sz(&mut self, qubits: &[usize]) -> &mut Self {
        self.sz_qubits.extend_from_slice(qubits);
        self
    }

    /// Adds CX gates to the buffer.
    ///
    /// Qubits should be pairs: [ctrl, tgt, ctrl, tgt, ...]
    #[inline]
    pub fn cx(&mut self, qubits: &[usize]) -> &mut Self {
        self.cx_qubits.extend_from_slice(qubits);
        self
    }

    /// Adds CZ gates to the buffer.
    #[inline]
    pub fn cz(&mut self, qubits: &[usize]) -> &mut Self {
        self.cz_qubits.extend_from_slice(qubits);
        self
    }

    /// Returns true if the buffer has no pending operations.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.h_qubits.is_empty()
            && self.x_qubits.is_empty()
            && self.z_qubits.is_empty()
            && self.sz_qubits.is_empty()
            && self.cx_qubits.is_empty()
            && self.cz_qubits.is_empty()
    }

    /// Clears all pending operations without executing them.
    #[inline]
    pub fn clear(&mut self) {
        self.h_qubits.clear();
        self.x_qubits.clear();
        self.z_qubits.clear();
        self.sz_qubits.clear();
        self.cx_qubits.clear();
        self.cz_qubits.clear();
    }

    /// Executes all accumulated operations on the simulator and clears the buffer.
    ///
    /// Operations are executed in an optimized order:
    /// 1. Single-qubit Pauli gates (X, Z) - pure phase updates
    /// 2. Single-qubit Clifford gates (H, SZ) - column operations
    /// 3. Two-qubit gates (CX, CZ) - cross-column operations
    pub fn flush<R: SeedableRng + Rng + Debug>(&mut self, sim: &mut SparseStabGeneric<BitSet, R>) {
        // Phase 1: Pauli gates (only affect signs)
        if !self.x_qubits.is_empty() {
            sim.x_batched(&self.x_qubits);
            self.x_qubits.clear();
        }
        if !self.z_qubits.is_empty() {
            sim.z_batched(&self.z_qubits);
            self.z_qubits.clear();
        }

        // Phase 2: Single-qubit Clifford gates
        if !self.sz_qubits.is_empty() {
            sim.sz_batched(&self.sz_qubits);
            self.sz_qubits.clear();
        }
        if !self.h_qubits.is_empty() {
            sim.h_batched(&self.h_qubits);
            self.h_qubits.clear();
        }

        // Phase 3: Two-qubit gates
        if !self.cx_qubits.is_empty() {
            sim.cx_batched(&self.cx_qubits);
            self.cx_qubits.clear();
        }
        if !self.cz_qubits.is_empty() {
            sim.cz_batched(&self.cz_qubits);
            self.cz_qubits.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CliffordGateable;
    use crate::SparseStab;
    use pecos_core::QubitId;

    #[test]
    fn test_h_batched_matches_sequential() {
        let mut sim1 = SparseStab::with_seed(10, 42);
        let mut sim2 = SparseStab::with_seed(10, 42);

        // Sequential H
        for q in 0..10 {
            sim1.h(&[QubitId(q)]);
        }

        // Batched H
        let qubits: Vec<usize> = (0..10).collect();
        sim2.h_batched(&qubits);

        // Compare states
        assert_eq!(format!("{:?}", sim1.stabs), format!("{:?}", sim2.stabs));
        assert_eq!(format!("{:?}", sim1.destabs), format!("{:?}", sim2.destabs));
    }

    #[test]
    fn test_x_batched_matches_sequential() {
        let mut sim1 = SparseStab::with_seed(10, 42);
        let mut sim2 = SparseStab::with_seed(10, 42);

        // Apply some H gates first to create non-trivial state
        for q in 0..5 {
            sim1.h(&[QubitId(q)]);
            sim2.h(&[QubitId(q)]);
        }

        // Sequential X
        for q in 0..10 {
            sim1.x(&[QubitId(q)]);
        }

        // Batched X
        let qubits: Vec<usize> = (0..10).collect();
        sim2.x_batched(&qubits);

        assert_eq!(format!("{:?}", sim1.stabs), format!("{:?}", sim2.stabs));
    }

    #[test]
    fn test_cx_batched_matches_sequential() {
        let mut sim1 = SparseStab::with_seed(8, 42);
        let mut sim2 = SparseStab::with_seed(8, 42);

        // Apply some H gates first
        for q in [0, 2, 4, 6] {
            sim1.h(&[QubitId(q)]);
            sim2.h(&[QubitId(q)]);
        }

        // Sequential CX
        sim1.cx(&[QubitId(0), QubitId(1)]);
        sim1.cx(&[QubitId(2), QubitId(3)]);
        sim1.cx(&[QubitId(4), QubitId(5)]);
        sim1.cx(&[QubitId(6), QubitId(7)]);

        // Batched CX
        sim2.cx_batched(&[0, 1, 2, 3, 4, 5, 6, 7]);

        assert_eq!(format!("{:?}", sim1.stabs), format!("{:?}", sim2.stabs));
        assert_eq!(format!("{:?}", sim1.destabs), format!("{:?}", sim2.destabs));
    }

    #[test]
    fn test_command_buffer_basic() {
        let mut sim1 = SparseStab::with_seed(8, 42);
        let mut sim2 = SparseStab::with_seed(8, 42);

        // Direct calls
        sim1.h(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        sim1.cx(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);

        // Via command buffer
        let mut buffer = CommandBuffer::new();
        buffer.h(&[0, 1, 2, 3]);
        buffer.cx(&[0, 1, 2, 3]);
        buffer.flush(&mut sim2);

        assert_eq!(format!("{:?}", sim1.stabs), format!("{:?}", sim2.stabs));
    }
}
