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

//! Sparse linear algebra over `Z_2` (GF(2)).
//!
//! Operations on binary vectors and matrices represented as sorted index
//! sets. This is the natural representation for QEC detector definitions
//! where each detector is an XOR (sum mod 2) of a small number of
//! measurements.
//!
//! # Representation
//!
//! A `Z_2` vector is a sorted `Vec<usize>` of indices where the vector has
//! value 1. Addition (XOR) is computed as sorted-merge symmetric difference.
//! A `Z_2` matrix is a `Vec<Vec<usize>>` of row vectors.
//!
//! # Example
//!
//! ```
//! use pecos_num::z2_linalg::{z2_rank, z2_xor};
//!
//! // Three detectors: D0=[0], D1=[1], D2=[0,1]
//! // D2 = D0 + D1 → rank should be 2
//! let rows = vec![vec![0], vec![1], vec![0, 1]];
//! assert_eq!(z2_rank(&rows), 2);
//!
//! // XOR of two sparse vectors
//! assert_eq!(z2_xor(&[1, 3, 5], &[2, 3, 4]), vec![1, 2, 4, 5]);
//! ```

use std::collections::BTreeMap;

/// Compute the rank of a binary matrix over `Z_2`.
///
/// Each row is a sorted list of column indices where the row has value 1.
/// Uses sparse Gaussian elimination with leftmost-column pivoting.
///
/// Complexity: O(n * k * log(n)) where n is the number of rows and k is
/// the average number of nonzeros per row. For QEC detectors with k ≈ 2,
/// this is effectively O(n * log(n)).
///
/// # Arguments
///
/// * `rows` - Binary matrix as a slice of sorted index vectors.
///
/// # Example
///
/// ```
/// use pecos_num::z2_linalg::z2_rank;
///
/// // Two independent rows
/// assert_eq!(z2_rank(&[vec![0], vec![1]]), 2);
///
/// // Three rows, one dependent (D2 = D0 + D1)
/// assert_eq!(z2_rank(&[vec![0, 1], vec![1, 2], vec![0, 2]]), 2);
/// ```
#[must_use]
pub fn z2_rank(rows: &[Vec<usize>]) -> usize {
    let mut work: Vec<Vec<usize>> = rows.to_vec();
    let mut pivot_rows: BTreeMap<usize, usize> = BTreeMap::new();
    let mut rank = 0;

    for i in 0..work.len() {
        // Reduce row i by XOR-ing with existing pivot rows
        loop {
            if work[i].is_empty() {
                break;
            }
            let min_col = work[i][0];
            if let Some(&pr) = pivot_rows.get(&min_col) {
                let pivot = work[pr].clone();
                work[i] = z2_xor(&work[i], &pivot);
            } else {
                break;
            }
        }

        if !work[i].is_empty() {
            let min_col = work[i][0];
            pivot_rows.insert(min_col, i);
            rank += 1;
        }
    }

    rank
}

/// Compute the rank of detector definitions given as record offsets.
///
/// Each record is a list of measurement indices (possibly negative for
/// Stim-style offsets from the end). Negative offsets are resolved against
/// `num_measurements`.
///
/// # Arguments
///
/// * `records` - Detector definitions as record offset lists.
/// * `num_measurements` - Total number of measurements (for resolving
///   negative offsets).
#[must_use]
pub fn z2_rank_from_records(records: &[Vec<i32>], num_measurements: usize) -> usize {
    let rows: Vec<Vec<usize>> = records
        .iter()
        .map(|record| {
            let mut indices: Vec<usize> = record
                .iter()
                .filter_map(|&offset| {
                    let abs = if offset < 0 {
                        num_measurements.checked_sub(offset.unsigned_abs() as usize)?
                    } else {
                        offset.unsigned_abs() as usize
                    };
                    (abs < num_measurements).then_some(abs)
                })
                .collect();
            indices.sort_unstable();
            indices.dedup();
            indices
        })
        .collect();

    z2_rank(&rows)
}

/// XOR (symmetric difference) of two sorted `Z_2` vectors.
///
/// Both inputs must be sorted and deduplicated. The result is also sorted
/// and deduplicated.
///
/// # Example
///
/// ```
/// use pecos_num::z2_linalg::z2_xor;
///
/// assert_eq!(z2_xor(&[1, 3, 5], &[2, 3, 4]), vec![1, 2, 4, 5]);
/// assert_eq!(z2_xor(&[0, 1], &[0, 1]), Vec::<usize>::new());
/// assert_eq!(z2_xor(&[], &[1, 2, 3]), vec![1, 2, 3]);
/// ```
#[must_use]
pub fn z2_xor(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }

    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

/// Check if a set of `Z_2` row vectors are linearly independent.
///
/// # Example
///
/// ```
/// use pecos_num::z2_linalg::z2_are_independent;
///
/// assert!(z2_are_independent(&[vec![0], vec![1]]));
/// assert!(!z2_are_independent(&[vec![0], vec![1], vec![0, 1]]));
/// ```
#[must_use]
pub fn z2_are_independent(rows: &[Vec<usize>]) -> bool {
    z2_rank(rows) == rows.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_empty() {
        assert_eq!(z2_rank(&[]), 0);
    }

    #[test]
    fn rank_single() {
        assert_eq!(z2_rank(&[vec![0]]), 1);
    }

    #[test]
    fn rank_independent() {
        assert_eq!(z2_rank(&[vec![0], vec![1], vec![2]]), 3);
    }

    #[test]
    fn rank_dependent() {
        assert_eq!(z2_rank(&[vec![0], vec![1], vec![0, 1]]), 2);
    }

    #[test]
    fn rank_all_identical() {
        assert_eq!(z2_rank(&[vec![0, 1], vec![0, 1], vec![0, 1]]), 1);
    }

    #[test]
    fn rank_chain_dependent() {
        // D0=[0,1], D1=[1,2], D2=[0,2] → D2 = D0 + D1 → rank 2
        assert_eq!(z2_rank(&[vec![0, 1], vec![1, 2], vec![0, 2]]), 2);
    }

    #[test]
    fn rank_large_sparse() {
        let rows: Vec<Vec<usize>> = (0..1000).map(|i| vec![i]).collect();
        assert_eq!(z2_rank(&rows), 1000);
    }

    #[test]
    fn rank_from_records_negative_offsets() {
        let records = vec![vec![-1i32], vec![-2]];
        assert_eq!(z2_rank_from_records(&records, 10), 2);
    }

    #[test]
    fn rank_from_records_dependent() {
        let records = vec![vec![0i32], vec![1], vec![0, 1]];
        assert_eq!(z2_rank_from_records(&records, 10), 2);
    }

    #[test]
    fn xor_basic() {
        assert_eq!(z2_xor(&[1, 3, 5], &[2, 3, 4]), vec![1, 2, 4, 5]);
    }

    #[test]
    fn xor_cancel() {
        assert_eq!(z2_xor(&[1, 2], &[1, 2]), Vec::<usize>::new());
    }

    #[test]
    fn xor_empty() {
        assert_eq!(z2_xor(&[], &[1, 2]), vec![1, 2]);
        assert_eq!(z2_xor(&[1, 2], &[]), vec![1, 2]);
    }

    #[test]
    fn independence_check() {
        assert!(z2_are_independent(&[vec![0], vec![1]]));
        assert!(!z2_are_independent(&[vec![0], vec![1], vec![0, 1]]));
        assert!(z2_are_independent(&[]));
    }
}
