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

//! GF(2) diagnostics for Optimization-Free Disentangling (OFD).
//!
//! Tracks the binary "flip pattern" of each non-Clifford gate applied to the STN.
//! The GF(2) rank of the accumulated flip matrix gives the theoretical minimum
//! bond dimension achievable by Clifford disentangling: `bond_dim` = 2^(t - rank),
//! where t is the number of non-Clifford gates.
//!
//! Based on: Liu & Clark, "Classical simulability of Clifford+T circuits with
//! CAMPS," arXiv:2412.17209 (2024).

/// Metadata associated with each non-Clifford gate tracked by the OFD matrix.
///
/// For OFD's fix-up Clifford construction, we need to know which qubit each
/// gate acted on. This lets a later "`in_span`" gate construct its absorption
/// Clifford from combinations of earlier gates' contributions.
#[derive(Clone, Copy, Debug)]
pub struct RowMetadata {
    /// The rotation axis qubit for the gate. For multi-site gates, this is
    /// the chosen `rot_site`; for single-site, it's the affected site.
    pub rot_site: usize,
}

/// GF(2) matrix tracking flip patterns from non-Clifford gate decompositions.
///
/// Each row is a binary vector of length `num_sites` (MPS sites). A 1 at position
/// j means the j-th destabilizer index was flipped (X or Y Pauli) in the
/// decomposition of `Z_q` for that non-Clifford gate.
#[derive(Clone, Debug)]
pub struct Gf2FlipMatrix {
    num_sites: usize,
    /// Rows stored as bit vectors (Vec<bool> for clarity; could use bitvec for perf).
    rows: Vec<Vec<bool>>,
    /// Metadata per row (parallel to `rows`). Populated by callers that want
    /// OFD fix-up info; left as default when only tracking rank.
    metadata: Vec<RowMetadata>,
}

impl Gf2FlipMatrix {
    /// Create an empty matrix for `num_sites` MPS sites.
    #[must_use]
    pub fn new(num_sites: usize) -> Self {
        Self {
            num_sites,
            rows: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// Add a row from a non-Clifford gate's decomposition.
    ///
    /// `flip_sites` are the destabilizer indices that have X or Y in the
    /// decomposition of `Z_q`. Metadata uses a default `rot_site` of 0 if not
    /// otherwise known; prefer `add_row_with_meta` for OFD work.
    pub fn add_row(&mut self, flip_sites: &[usize]) {
        self.add_row_with_meta(flip_sites, RowMetadata { rot_site: 0 });
    }

    /// Add a row with explicit metadata for OFD fix-up construction.
    pub fn add_row_with_meta(&mut self, flip_sites: &[usize], meta: RowMetadata) {
        let mut row = vec![false; self.num_sites];
        for &site in flip_sites {
            if site < self.num_sites {
                row[site] = true;
            }
        }
        self.rows.push(row);
        self.metadata.push(meta);
    }

    /// Metadata for row `i`, if it exists.
    #[must_use]
    pub fn row_metadata(&self, i: usize) -> Option<RowMetadata> {
        self.metadata.get(i).copied()
    }

    /// Number of non-Clifford gates tracked.
    #[must_use]
    pub fn num_gates(&self) -> usize {
        self.rows.len()
    }

    /// Compute the GF(2) rank via Gaussian elimination.
    ///
    /// Returns the rank (number of linearly independent rows over GF(2)).
    #[must_use]
    pub fn gf2_rank(&self) -> usize {
        if self.rows.is_empty() {
            return 0;
        }

        // Work on a copy for row reduction
        let mut matrix: Vec<Vec<bool>> = self.rows.clone();
        let num_rows = matrix.len();
        let num_cols = self.num_sites;

        let mut current_row = 0;

        // Standard GF(2) Gaussian elimination: sweep over columns.
        // Row pointer only advances when a pivot is found.
        for col in 0..num_cols {
            if current_row >= num_rows {
                break;
            }

            // Find a row with a 1 in this column at or below current_row
            let found = matrix[current_row..]
                .iter()
                .position(|row| row[col])
                .map(|offset| current_row + offset);

            if let Some(swap_row) = found {
                matrix.swap(current_row, swap_row);

                // Eliminate all other 1s in this column.
                // We need to XOR the pivot row into other rows, so split
                // into slices to avoid double-borrow.
                let pivot_row = matrix[current_row].clone();
                for (r, row) in matrix.iter_mut().enumerate() {
                    if r != current_row && row[col] {
                        for (cell, &piv) in row.iter_mut().zip(pivot_row.iter()) {
                            *cell ^= piv;
                        }
                    }
                }

                current_row += 1;
            }
        }

        current_row // = rank
    }

    /// Theoretical minimum bond dimension achievable by Clifford disentangling.
    ///
    /// When all non-Clifford gates' flip patterns are linearly independent over
    /// GF(2), each can be disentangled to a single site (bond dim stays 1).
    /// When there are dependencies, each dependency doubles the bond dim.
    ///
    /// Returns `2^(num_gates - rank)`.
    #[must_use]
    pub fn theoretical_min_bond_dim(&self) -> usize {
        let t = self.num_gates();
        let r = self.gf2_rank();
        if t <= r { 1 } else { 1 << (t - r) }
    }

    /// Reset the matrix (e.g., after simulator reset).
    pub fn reset(&mut self) {
        self.rows.clear();
        self.metadata.clear();
    }

    /// Check whether a new flip row is in the span of already-added rows.
    ///
    /// Returns `true` if adding this row would NOT increase the GF(2) rank,
    /// meaning the corresponding non-Clifford gate can be implemented using
    /// flip patterns already tracked (zero bond-dim growth).
    #[must_use]
    pub fn is_in_span(&self, new_row: &[usize]) -> bool {
        self.span_decomposition(new_row).is_some()
    }

    /// Find the linear combination of existing rows whose XOR equals `new_row`.
    ///
    /// Returns `Some(indices)` if `new_row` is in the span, where `indices`
    /// are original row indices whose XOR equals `new_row`. Returns `None`
    /// if `new_row` is linearly independent (would grow rank).
    ///
    /// Algorithm: augment the matrix with identity (tracking which original
    /// rows contribute), perform row-reduction, then reduce the target row
    /// against the augmented basis.
    #[must_use]
    pub fn span_decomposition(&self, new_row: &[usize]) -> Option<Vec<usize>> {
        let num_rows = self.rows.len();
        let num_cols = self.num_sites;
        // Augmented matrix: each row is (original row bits, provenance bits).
        // provenance[i] tracks which original rows have been XORed into this row.
        let mut aug: Vec<(Vec<bool>, Vec<bool>)> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let mut prov = vec![false; num_rows];
                prov[i] = true;
                (r.clone(), prov)
            })
            .collect();

        // Gaussian eliminate to RREF, maintaining provenance.
        let mut current_row = 0;
        for col in 0..num_cols {
            if current_row >= num_rows {
                break;
            }
            let found = aug[current_row..]
                .iter()
                .position(|entry| entry.0[col])
                .map(|offset| current_row + offset);
            if let Some(sw) = found {
                aug.swap(current_row, sw);
                let pivot_data = aug[current_row].0.clone();
                let pivot_prov = aug[current_row].1.clone();
                for (r, entry) in aug.iter_mut().enumerate() {
                    if r != current_row && entry.0[col] {
                        // XOR current_row into r (both data and provenance).
                        for (cell, &piv) in entry.0.iter_mut().zip(pivot_data.iter()) {
                            *cell ^= piv;
                        }
                        for (cell, &piv) in entry.1.iter_mut().zip(pivot_prov.iter()) {
                            *cell ^= piv;
                        }
                    }
                }
                current_row += 1;
            }
        }

        // Build target vector.
        let mut v = vec![false; num_cols];
        for &s in new_row {
            if s < num_cols {
                v[s] = true;
            }
        }
        let mut combination = vec![false; num_rows];

        // Reduce v against RREF basis, accumulating provenance.
        for entry in &aug[..current_row] {
            if let Some(pivot) = entry.0.iter().position(|&b| b)
                && v[pivot]
            {
                for (vc, &ec) in v.iter_mut().zip(entry.0.iter()) {
                    *vc ^= ec;
                }
                for (cc, &ep) in combination.iter_mut().zip(entry.1.iter()) {
                    *cc ^= ep;
                }
            }
        }

        // If v is all-zero, the new_row was in span; combination gives the decomposition.
        if v.iter().all(|&b| !b) {
            Some(
                combination
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &b)| if b { Some(i) } else { None })
                    .collect(),
            )
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_matrix() {
        let m = Gf2FlipMatrix::new(4);
        assert_eq!(m.gf2_rank(), 0);
        assert_eq!(m.theoretical_min_bond_dim(), 1);
    }

    #[test]
    fn test_single_row() {
        let mut m = Gf2FlipMatrix::new(4);
        m.add_row(&[0, 2]); // flip sites 0 and 2
        assert_eq!(m.gf2_rank(), 1);
        assert_eq!(m.theoretical_min_bond_dim(), 1); // 2^(1-1) = 1
    }

    #[test]
    fn test_two_independent_rows() {
        let mut m = Gf2FlipMatrix::new(4);
        m.add_row(&[0]); // [1,0,0,0]
        m.add_row(&[1]); // [0,1,0,0]
        assert_eq!(m.gf2_rank(), 2);
        assert_eq!(m.theoretical_min_bond_dim(), 1); // 2^(2-2) = 1
    }

    #[test]
    fn test_two_dependent_rows() {
        let mut m = Gf2FlipMatrix::new(4);
        m.add_row(&[0, 1]); // [1,1,0,0]
        m.add_row(&[0, 1]); // [1,1,0,0] -- same row
        assert_eq!(m.gf2_rank(), 1);
        assert_eq!(m.theoretical_min_bond_dim(), 2); // 2^(2-1) = 2
    }

    #[test]
    fn test_three_rows_one_dependent() {
        let mut m = Gf2FlipMatrix::new(4);
        m.add_row(&[0, 1]); // [1,1,0,0]
        m.add_row(&[1, 2]); // [0,1,1,0]
        m.add_row(&[0, 2]); // [1,0,1,0] = row1 XOR row2
        assert_eq!(m.gf2_rank(), 2);
        assert_eq!(m.theoretical_min_bond_dim(), 2); // 2^(3-2) = 2
    }

    #[test]
    fn test_full_rank_n_equals_t() {
        // 4 independent rows in 4 columns = rank 4
        let mut m = Gf2FlipMatrix::new(4);
        m.add_row(&[0]);
        m.add_row(&[1]);
        m.add_row(&[2]);
        m.add_row(&[3]);
        assert_eq!(m.gf2_rank(), 4);
        assert_eq!(m.theoretical_min_bond_dim(), 1);
    }

    #[test]
    fn test_is_in_span_empty() {
        let m = Gf2FlipMatrix::new(3);
        // Empty basis -- only zero vector is in span.
        assert!(m.is_in_span(&[])); // all-zero row is always in span (trivially)
        assert!(!m.is_in_span(&[0]));
        assert!(!m.is_in_span(&[1, 2]));
    }

    #[test]
    fn test_is_in_span_single_row() {
        let mut m = Gf2FlipMatrix::new(3);
        m.add_row(&[0]); // basis: {e_0}
        assert!(m.is_in_span(&[0]));
        assert!(!m.is_in_span(&[1]));
        assert!(!m.is_in_span(&[0, 1])); // e_0 + e_1 not in span of {e_0}
    }

    #[test]
    fn test_is_in_span_dependency() {
        let mut m = Gf2FlipMatrix::new(3);
        m.add_row(&[0]);
        m.add_row(&[1]);
        // Now {e_0, e_1} basis. e_0 XOR e_1 = (1,1,0) is in span.
        assert!(m.is_in_span(&[0, 1]));
        // e_2 is NOT in span.
        assert!(!m.is_in_span(&[2]));
        // e_0 XOR e_1 XOR e_2 is NOT in span (needs e_2).
        assert!(!m.is_in_span(&[0, 1, 2]));
    }

    #[test]
    fn test_span_decomposition_simple() {
        let mut m = Gf2FlipMatrix::new(3);
        m.add_row(&[0]); // row 0: e_0
        m.add_row(&[1]); // row 1: e_1
        // e_0 + e_1 = (1,1,0) should decompose to {0, 1}.
        let dep = m.span_decomposition(&[0, 1]).expect("in span");
        assert_eq!(dep, vec![0, 1]);
        // e_0 alone decomposes to {0}.
        let dep = m.span_decomposition(&[0]).expect("in span");
        assert_eq!(dep, vec![0]);
        // e_2 is not in span.
        assert!(m.span_decomposition(&[2]).is_none());
    }

    #[test]
    fn test_span_decomposition_verify_xor() {
        // Property: the returned indices XOR to the input row.
        let mut m = Gf2FlipMatrix::new(5);
        m.add_row(&[0, 1]);
        m.add_row(&[2, 3]);
        m.add_row(&[1, 3, 4]);
        m.add_row(&[0, 2, 4]); // Should be dependent: row0 XOR row1 XOR row2 = (1,1,0,0,0) XOR (0,0,1,1,0) XOR (0,1,0,1,1) = (1,0,1,0,1)
        // Test that (1,0,1,0,1) decomposes properly.
        let target = &[0, 2, 4];
        let dep = m.span_decomposition(target).expect("should be in span");
        // Verify the XOR reconstructs target.
        let mut recon = vec![false; 5];
        for &i in &dep {
            for (rc, &rv) in recon.iter_mut().zip(m.rows[i].iter()) {
                *rc ^= rv;
            }
        }
        let mut target_vec = vec![false; 5];
        for &s in target {
            target_vec[s] = true;
        }
        assert_eq!(recon, target_vec, "XOR of rows {dep:?} should equal target");
    }

    #[test]
    fn test_is_in_span_matches_rank_check() {
        // Property: is_in_span(row) iff adding row doesn't change rank.
        let mut m = Gf2FlipMatrix::new(4);
        m.add_row(&[0, 1]);
        m.add_row(&[2, 3]);
        m.add_row(&[0, 2]);
        let rank_before = m.gf2_rank();
        for row in [
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![0, 1],
            vec![1, 2],
            vec![0, 1, 2, 3],
        ] {
            let in_span = m.is_in_span(&row);
            let mut m2 = m.clone();
            m2.add_row(&row);
            let rank_after = m2.gf2_rank();
            assert_eq!(
                in_span,
                rank_after == rank_before,
                "row {row:?}: is_in_span={in_span} but rank {rank_before} -> {rank_after}"
            );
        }
    }

    #[test]
    fn test_more_rows_than_cols() {
        // 5 rows, 3 cols -> rank <= 3, so at least 2 dependencies
        let mut m = Gf2FlipMatrix::new(3);
        m.add_row(&[0]);
        m.add_row(&[1]);
        m.add_row(&[2]);
        m.add_row(&[0, 1]); // dependent: row1 XOR row2
        m.add_row(&[1, 2]); // dependent: row2 XOR row3
        assert_eq!(m.gf2_rank(), 3);
        assert_eq!(m.theoretical_min_bond_dim(), 4); // 2^(5-3) = 4
    }
}
