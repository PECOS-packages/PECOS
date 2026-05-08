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
///
/// Rows are stored as `u128` bitmasks for allocation-free GF(2) operations.
/// Supports up to 128 MPS sites.
#[derive(Clone, Debug)]
pub struct Gf2FlipMatrix {
    num_sites: usize,
    /// Rows stored as u128 bitmasks. Bit j is set iff site j is flipped.
    rows: Vec<u128>,
    /// Metadata per row (parallel to `rows`).
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
        let mut row: u128 = 0;
        for &site in flip_sites {
            if site < self.num_sites && site < 128 {
                row |= 1u128 << site;
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

        let mut matrix: Vec<u128> = self.rows.clone();
        let num_rows = matrix.len();
        let cols = self.num_sites.min(128);

        let mut current_row = 0;
        for col in 0..cols {
            if current_row >= num_rows {
                break;
            }
            let col_bit = 1u128 << col;
            let found = matrix[current_row..]
                .iter()
                .position(|&row| row & col_bit != 0)
                .map(|offset| current_row + offset);
            if let Some(swap_row) = found {
                matrix.swap(current_row, swap_row);
                let pivot = matrix[current_row];
                for (r, row) in matrix.iter_mut().enumerate() {
                    if r != current_row && *row & col_bit != 0 {
                        *row ^= pivot;
                    }
                }
                current_row += 1;
            }
        }
        current_row
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
    /// Uses u128 bitmasks for both data and provenance — zero heap allocation.
    #[must_use]
    pub fn span_decomposition(&self, new_row: &[usize]) -> Option<Vec<usize>> {
        let num_rows = self.rows.len();
        if num_rows == 0 {
            // Empty matrix: new_row is in span only if it's zero.
            let mut target: u128 = 0;
            for &s in new_row {
                if s < self.num_sites && s < 128 { target |= 1u128 << s; }
            }
            return if target == 0 { Some(Vec::new()) } else { None };
        }

        // Augmented rows: (data bits, provenance bits).
        // Provenance uses u128 bitmask — supports up to 64 accumulated rows.
        // For larger matrices, fall back to Vec-based provenance.
        if num_rows <= 128 {
            return self.span_decomposition_fast(new_row);
        }

        // Fallback for >64 rows (unlikely in practice)
        self.span_decomposition_large(new_row)
    }

    /// Fast span decomposition using u128 bitmasks for both data and provenance.
    fn span_decomposition_fast(&self, new_row: &[usize]) -> Option<Vec<usize>> {
        let num_rows = self.rows.len();

        // Augmented: (data_bits, provenance_bits) — all stack-allocated.
        let mut data: Vec<u128> = self.rows.clone();
        let mut prov: Vec<u128> = (0..num_rows).map(|i| 1u128 << i).collect();

        // Gaussian elimination to RREF.
        let cols = self.num_sites.min(128);
        let mut current_row = 0;
        for col in 0..cols {
            if current_row >= num_rows {
                break;
            }
            let col_bit = 1u128 << col;
            let found = data[current_row..]
                .iter()
                .position(|&d| d & col_bit != 0)
                .map(|offset| current_row + offset);
            if let Some(sw) = found {
                data.swap(current_row, sw);
                prov.swap(current_row, sw);
                let pivot_d = data[current_row];
                let pivot_p = prov[current_row];
                for r in 0..num_rows {
                    if r != current_row && data[r] & col_bit != 0 {
                        data[r] ^= pivot_d;
                        prov[r] ^= pivot_p;
                    }
                }
                current_row += 1;
            }
        }

        // Build target and reduce against RREF basis.
        let mut v: u128 = 0;
        for &s in new_row {
            if s < self.num_sites && s < 128 { v |= 1u128 << s; }
        }
        let mut combination: u128 = 0;

        for i in 0..current_row {
            let pivot = data[i].trailing_zeros() as usize;
            if pivot < self.num_sites && v & (1u128 << pivot) != 0 {
                v ^= data[i];
                combination ^= prov[i];
            }
        }

        if v == 0 {
            Some(
                (0..num_rows)
                    .filter(|&i| combination & (1u128 << i) != 0)
                    .collect(),
            )
        } else {
            None
        }
    }

    /// Fallback span decomposition for >64 rows.
    fn span_decomposition_large(&self, new_row: &[usize]) -> Option<Vec<usize>> {
        let num_rows = self.rows.len();
        let mut data: Vec<u128> = self.rows.clone();
        let mut prov: Vec<Vec<bool>> = (0..num_rows)
            .map(|i| {
                let mut p = vec![false; num_rows];
                p[i] = true;
                p
            })
            .collect();

        let cols = self.num_sites.min(128);
        let mut current_row = 0;
        for col in 0..cols {
            if current_row >= num_rows {
                break;
            }
            let col_bit = 1u128 << col;
            let found = data[current_row..]
                .iter()
                .position(|&d| d & col_bit != 0)
                .map(|offset| current_row + offset);
            if let Some(sw) = found {
                data.swap(current_row, sw);
                prov.swap(current_row, sw);
                let pivot_d = data[current_row];
                let pivot_p = prov[current_row].clone();
                for r in 0..num_rows {
                    if r != current_row && data[r] & col_bit != 0 {
                        data[r] ^= pivot_d;
                        for (c, &p) in prov[r].iter_mut().zip(pivot_p.iter()) {
                            *c ^= p;
                        }
                    }
                }
                current_row += 1;
            }
        }

        let mut v: u128 = 0;
        for &s in new_row {
            if s < self.num_sites && s < 128 { v |= 1u128 << s; }
        }
        let mut combination = vec![false; num_rows];
        for i in 0..current_row {
            let pivot = data[i].trailing_zeros() as usize;
            if pivot < self.num_sites && v & (1u128 << pivot) != 0 {
                v ^= data[i];
                for (c, &p) in combination.iter_mut().zip(prov[i].iter()) {
                    *c ^= p;
                }
            }
        }

        if v == 0 {
            Some(
                combination.iter().enumerate()
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
        let mut recon: u128 = 0;
        for &i in &dep {
            recon ^= m.rows[i];
        }
        let mut target_bits: u128 = 0;
        for &s in target {
            target_bits |= 1u128 << s;
        }
        assert_eq!(recon, target_bits, "XOR of rows {dep:?} should equal target");
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
