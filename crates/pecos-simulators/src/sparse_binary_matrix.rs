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

//! A sparse binary matrix with dual row/column representation.
//!
//! This provides O(weight) row XOR operations while maintaining O(1) column membership
//! queries, using the same dual-representation trick as [`GensGeneric`](crate::gens::GensGeneric).
//!
//! Used by the CH-form simulator for its F, G, M matrices.

use pecos_core::{BitSet, IndexSet};

/// A square binary matrix stored in both row-wise and column-wise sparse form.
///
/// The dual representation allows efficient row operations (XOR, swap) while
/// also providing fast column access for operations like inner products.
///
/// # Invariant
/// For all i, j: `rows[i].contains(j) == cols[j].contains(i)`
#[derive(Clone, Debug)]
pub struct SparseBinaryMatrix<S: IndexSet = BitSet> {
    n: usize,
    rows: Vec<S>,
    cols: Vec<S>,
}

impl<S: IndexSet> SparseBinaryMatrix<S> {
    /// Create an n x n zero matrix.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let rows = (0..n).map(|_| S::with_capacity(n)).collect();
        let cols = (0..n).map(|_| S::with_capacity(n)).collect();
        Self { n, rows, cols }
    }

    /// Create an n x n identity matrix.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let mut mat = Self::new(n);
        for i in 0..n {
            mat.rows[i].insert(i);
            mat.cols[i].insert(i);
        }
        mat
    }

    /// Matrix dimension (n for an n x n matrix).
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Get M[i][j].
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> bool {
        self.rows[i].contains(j)
    }

    /// Set M[i][j] = val.
    pub fn set(&mut self, i: usize, j: usize, val: bool) {
        if val {
            self.rows[i].insert(j);
            self.cols[j].insert(i);
        } else {
            self.rows[i].remove(j);
            self.cols[j].remove(i);
        }
    }

    /// Toggle M[i][j].
    pub fn toggle(&mut self, i: usize, j: usize) {
        self.rows[i].toggle(j);
        self.cols[j].toggle(i);
    }

    /// XOR row `src` into row `dst`: rows[dst] ^= rows[src].
    ///
    /// Updates both row and column representations. `dst` must not equal `src`.
    pub fn row_xor_assign(&mut self, dst: usize, src: usize) {
        debug_assert_ne!(dst, src, "row_xor_assign: dst must differ from src");

        // Update columns first: for each column j in src's row, toggle dst in that column.
        for j in self.rows[src].iter() {
            self.cols[j].toggle(dst);
        }

        // Update the row. Need unsafe because we borrow rows[dst] mutably and rows[src] immutably.
        let src_ptr = std::ptr::from_ref(&self.rows[src]);
        unsafe {
            self.rows[dst].xor_assign(&*src_ptr);
        }

        debug_assert_consistent(self);
    }

    /// XOR a row from another matrix into row `dst` of this matrix:
    /// `self.rows[dst] ^= other.rows[src]`.
    ///
    /// This is used for cross-matrix operations in CH-form (e.g., M[q,:] ^= G[r,:]).
    pub fn row_xor_from(&mut self, dst: usize, other: &Self, src: usize) {
        // Update columns: for each column j in other's src row, toggle dst in our column j
        for j in other.rows[src].iter() {
            self.cols[j].toggle(dst);
        }

        // Update the row
        self.rows[dst].xor_assign(&other.rows[src]);

        debug_assert_consistent(self);
    }

    /// Swap rows i and j (updating both representations).
    pub fn swap_rows(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        self.rows.swap(i, j);
        // Update column sets: everywhere i appeared, replace with j and vice versa
        for col_k in &mut self.cols {
            let has_i = col_k.contains(i);
            let has_j = col_k.contains(j);
            if has_i != has_j {
                col_k.toggle(i);
                col_k.toggle(j);
            }
        }
        debug_assert_consistent(self);
    }

    /// Swap columns i and j (updating both representations).
    pub fn swap_cols(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        self.cols.swap(i, j);
        // Update row sets
        for row_k in &mut self.rows {
            let has_i = row_k.contains(i);
            let has_j = row_k.contains(j);
            if has_i != has_j {
                row_k.toggle(i);
                row_k.toggle(j);
            }
        }
        debug_assert_consistent(self);
    }

    /// XOR a set into row `dst`: `self.rows[dst] ^= set`.
    ///
    /// Useful when the source set is computed externally.
    pub fn row_xor_set(&mut self, dst: usize, set: &S) {
        for j in set.iter() {
            self.cols[j].toggle(dst);
        }
        self.rows[dst].xor_assign(set);
        debug_assert_consistent(self);
    }

    /// Count of positions where both row i and row j have a 1.
    ///
    /// Returns the count (not reduced mod 2). Use `& 1` for the mod-2 inner product.
    #[inline]
    #[must_use]
    pub fn row_inner_product(&self, i: usize, j: usize) -> usize {
        self.rows[i].intersection_count(&self.rows[j])
    }

    /// Access row i as a set reference.
    #[inline]
    #[must_use]
    pub fn row(&self, i: usize) -> &S {
        &self.rows[i]
    }

    /// Access column j as a set reference.
    #[inline]
    #[must_use]
    pub fn col(&self, j: usize) -> &S {
        &self.cols[j]
    }

    /// Reset to zero matrix.
    pub fn reset_to_zero(&mut self) {
        for r in &mut self.rows {
            r.clear();
        }
        for c in &mut self.cols {
            c.clear();
        }
    }

    /// Reset to identity matrix.
    pub fn reset_to_identity(&mut self) {
        for (i, r) in self.rows.iter_mut().enumerate() {
            r.set_single(i);
        }
        for (j, c) in self.cols.iter_mut().enumerate() {
            c.set_single(j);
        }
    }
}

/// Verify the row-column consistency invariant (debug builds only).
#[inline]
fn debug_assert_consistent<S: IndexSet>(mat: &SparseBinaryMatrix<S>) {
    if cfg!(debug_assertions) {
        for i in 0..mat.n {
            for j in mat.rows[i].iter() {
                debug_assert!(
                    mat.cols[j].contains(i),
                    "Row-column inconsistency: rows[{i}] has {j} but cols[{j}] missing {i}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::BitSet;

    type Mat = SparseBinaryMatrix<BitSet>;

    #[test]
    fn test_new_is_zero() {
        let m = Mat::new(4);
        for i in 0..4 {
            for j in 0..4 {
                assert!(!m.get(i, j));
            }
        }
    }

    #[test]
    fn test_identity() {
        let m = Mat::identity(4);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(m.get(i, j), i == j, "identity[{i}][{j}]");
            }
        }
    }

    #[test]
    fn test_set_and_get() {
        let mut m = Mat::new(3);
        m.set(0, 2, true);
        m.set(1, 1, true);
        assert!(m.get(0, 2));
        assert!(m.get(1, 1));
        assert!(!m.get(0, 0));

        // Column view consistent
        assert!(m.col(2).contains(0));
        assert!(m.col(1).contains(1));
        assert!(!m.col(0).contains(0));

        // Unset
        m.set(0, 2, false);
        assert!(!m.get(0, 2));
        assert!(!m.col(2).contains(0));
    }

    #[test]
    fn test_toggle() {
        let mut m = Mat::new(3);
        m.toggle(1, 2);
        assert!(m.get(1, 2));
        m.toggle(1, 2);
        assert!(!m.get(1, 2));
    }

    #[test]
    fn test_row_xor_assign() {
        // Start with identity
        let mut m = Mat::identity(3);
        // rows: [0]={0}, [1]={1}, [2]={2}

        // XOR row 1 into row 0: row[0] = {0} ^ {1} = {0,1}
        m.row_xor_assign(0, 1);
        assert!(m.get(0, 0));
        assert!(m.get(0, 1));
        assert!(!m.get(0, 2));

        // Column consistency
        assert!(m.col(0).contains(0));
        assert!(m.col(1).contains(0));
        assert!(m.col(1).contains(1));
    }

    #[test]
    fn test_row_xor_assign_cancellation() {
        let mut m = Mat::identity(3);
        // XOR row 0 into row 1 twice -> should cancel out
        m.row_xor_assign(1, 0);
        m.row_xor_assign(1, 0);
        // row[1] should be back to {1}
        assert!(!m.get(1, 0));
        assert!(m.get(1, 1));
        assert!(!m.get(1, 2));
    }

    #[test]
    fn test_swap_rows() {
        let mut m = Mat::identity(3);
        m.set(0, 2, true); // row[0] = {0, 2}
        m.swap_rows(0, 1);
        // row[0] should now be old row[1] = {1}
        assert!(!m.get(0, 0));
        assert!(m.get(0, 1));
        assert!(!m.get(0, 2));
        // row[1] should now be old row[0] = {0, 2}
        assert!(m.get(1, 0));
        assert!(!m.get(1, 1));
        assert!(m.get(1, 2));
    }

    #[test]
    fn test_swap_cols() {
        let mut m = Mat::identity(3);
        m.set(0, 2, true); // row[0] = {0, 2}
        m.swap_cols(0, 2);
        // col 0 and col 2 swapped:
        // row[0] was {0, 2} -> after swap: {2, 0} = {0, 2}... wait.
        // Actually, swapping columns 0 and 2 in [0,0]=1, [0,2]=1 gives [0,2]=1, [0,0]=1
        // which is the same! Let's test a case where it actually changes.
        let mut m2 = Mat::new(3);
        m2.set(0, 0, true); // row[0] = {0}
        m2.set(1, 2, true); // row[1] = {2}
        m2.swap_cols(0, 2);
        // After swap: [0,0] was 1 -> now [0,2] = 1; [1,2] was 1 -> now [1,0] = 1
        assert!(!m2.get(0, 0));
        assert!(m2.get(0, 2));
        assert!(m2.get(1, 0));
        assert!(!m2.get(1, 2));
    }

    #[test]
    fn test_row_inner_product() {
        let mut m = Mat::new(4);
        m.set(0, 0, true);
        m.set(0, 1, true);
        m.set(0, 3, true); // row[0] = {0, 1, 3}
        m.set(1, 1, true);
        m.set(1, 2, true);
        m.set(1, 3, true); // row[1] = {1, 2, 3}
        // intersection = {1, 3}, count = 2
        assert_eq!(m.row_inner_product(0, 1), 2);
        assert_eq!(m.row_inner_product(0, 1) & 1, 0); // mod 2 = 0
    }

    #[test]
    fn test_reset_to_zero() {
        let mut m = Mat::identity(3);
        m.reset_to_zero();
        for i in 0..3 {
            for j in 0..3 {
                assert!(!m.get(i, j));
            }
        }
    }

    #[test]
    fn test_reset_to_identity() {
        let mut m = Mat::new(3);
        m.set(0, 1, true);
        m.set(2, 0, true);
        m.reset_to_identity();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(m.get(i, j), i == j);
            }
        }
    }

    #[test]
    fn test_row_xor_from() {
        let mut m = Mat::new(3);
        let g = Mat::identity(3);
        // m.row(0) ^= g.row(1): m[0] = {} ^ {1} = {1}
        m.row_xor_from(0, &g, 1);
        assert!(m.get(0, 1));
        assert!(!m.get(0, 0));
        // m.row(0) ^= g.row(2): m[0] = {1} ^ {2} = {1, 2}
        m.row_xor_from(0, &g, 2);
        assert!(m.get(0, 1));
        assert!(m.get(0, 2));
        // Column consistency
        assert!(m.col(1).contains(0));
        assert!(m.col(2).contains(0));
    }

    #[test]
    fn test_complex_operations_stay_consistent() {
        // A sequence of mixed operations, checking consistency throughout
        let mut m = Mat::identity(4);
        m.row_xor_assign(0, 1); // row[0] = {0, 1}
        m.row_xor_assign(0, 2); // row[0] = {0, 1, 2}
        m.set(3, 0, true); // row[3] = {0, 3}
        m.row_xor_assign(3, 0); // row[3] = {0, 3} ^ {0, 1, 2} = {1, 2, 3}
        m.swap_rows(1, 2);
        m.toggle(0, 0); // row[0] = {1, 2} (removed 0)
        m.swap_cols(1, 3);

        // Just verify the invariant holds
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    m.get(i, j),
                    m.col(j).contains(i),
                    "Inconsistency at [{i}][{j}]"
                );
            }
        }
    }
}
