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

//! A bit-vector based set of `usize` indices.
//!
//! This module provides [`BitSet`], a compact set implementation optimized for
//! storing sets of small non-negative integers (indices).
//!
//! # Key Features
//!
//! - **O(1) insert/remove/contains** - constant time membership operations
//! - **O(words) XOR** - symmetric difference via bitwise operations (where words = `max_index/64`)
//! - **Compact storage** - 1 bit per possible index
//! - **Fast iteration** - uses hardware popcount for efficient traversal
//!
//! # Use Cases
//!
//! - Tracking measurement indices in symbolic stabilizer simulation
//! - Sparse row/column indices in linear algebra
//! - Any scenario where XOR (symmetric difference) is the dominant operation
//!
//! # Example
//!
//! ```rust
//! use pecos_core::BitSet;
//!
//! let mut set = BitSet::new();
//! set.insert(0);
//! set.insert(5);
//! set.insert(100);
//!
//! assert!(set.contains(5));
//! assert!(!set.contains(6));
//!
//! // Fast XOR operation
//! let mut other = BitSet::single(5);
//! set ^= &other;  // {0, 100} - the 5 cancels out
//! ```

use std::collections::BTreeSet;

/// A bit-vector based set of `usize` indices.
///
/// Uses a `Vec<u64>` internally where each bit represents membership.
/// Index `i` is stored in word `i / 64`, bit position `i % 64`.
///
/// # Performance
///
/// | Operation | Time Complexity |
/// |-----------|-----------------|
/// | insert    | O(1) amortized  |
/// | remove    | O(1)            |
/// | contains  | O(1)            |
/// | XOR       | O(words) where words = `max_index/64` |
/// | `is_empty`  | O(words)        |
/// | len       | O(words)        |
/// | iter      | O(n + words)    |
///
/// For typical use cases (indices 0-5000), this is very efficient.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct BitSet {
    /// Bit vector storage: word `i` contains bits for indices `i*64` to `i*64+63`
    words: Vec<u64>,
}

impl BitSet {
    /// Create an empty set.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self { words: Vec::new() }
    }

    /// Create a set with capacity for at least `max_index` indices.
    #[inline]
    #[must_use]
    pub fn with_capacity(max_index: usize) -> Self {
        let num_words = max_index.div_ceil(64);
        Self {
            words: vec![0u64; num_words],
        }
    }

    /// Create a set containing a single index.
    #[inline]
    #[must_use]
    pub fn single(index: usize) -> Self {
        let mut set = Self::new();
        set.insert(index);
        set
    }

    /// Insert an index into the set.
    ///
    /// Returns `true` if the index was newly inserted, `false` if it was already present.
    #[inline]
    pub fn insert(&mut self, index: usize) -> bool {
        let word_idx = index / 64;
        let bit_idx = index % 64;
        let mask = 1u64 << bit_idx;

        // Extend if necessary
        if word_idx >= self.words.len() {
            self.words.resize(word_idx + 1, 0);
        }

        let was_present = (self.words[word_idx] & mask) != 0;
        self.words[word_idx] |= mask;
        !was_present
    }

    /// Remove an index from the set.
    ///
    /// Returns `true` if the index was present, `false` otherwise.
    #[inline]
    pub fn remove(&mut self, index: usize) -> bool {
        let word_idx = index / 64;
        let bit_idx = index % 64;

        if word_idx >= self.words.len() {
            return false;
        }

        let mask = 1u64 << bit_idx;
        let was_present = (self.words[word_idx] & mask) != 0;
        self.words[word_idx] &= !mask;
        was_present
    }

    /// Check if the set contains an index.
    #[inline]
    #[must_use]
    pub fn contains(&self, index: usize) -> bool {
        let word_idx = index / 64;
        let bit_idx = index % 64;

        if word_idx >= self.words.len() {
            return false;
        }

        (self.words[word_idx] & (1u64 << bit_idx)) != 0
    }

    /// Check if the set is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Returns the number of elements in the set.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Clear all elements from the set.
    #[inline]
    pub fn clear(&mut self) {
        for w in &mut self.words {
            *w = 0;
        }
    }

    /// XOR (symmetric difference) with another set, in place.
    ///
    /// Elements present in exactly one of the two sets will be in the result.
    /// This is the primary operation for measurement index tracking.
    #[inline]
    pub fn symmetric_difference_update(&mut self, other: &Self) {
        // Extend if other is longer
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }

        // XOR the overlapping portion
        for (self_word, &other_word) in self.words.iter_mut().zip(other.words.iter()) {
            *self_word ^= other_word;
        }
    }

    /// Returns a new set that is the XOR of this set and another.
    #[inline]
    #[must_use]
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.symmetric_difference_update(other);
        result
    }

    /// Iterate over the indices in the set (in ascending order).
    #[inline]
    #[must_use]
    pub fn iter(&self) -> BitSetIter<'_> {
        BitSetIter {
            words: &self.words,
            word_idx: 0,
            current_word: self.words.first().copied().unwrap_or(0),
            base_index: 0,
        }
    }

    /// Convert to a `BTreeSet<usize>` for compatibility with existing code.
    #[must_use]
    pub fn to_btree_set(&self) -> BTreeSet<usize> {
        self.iter().collect()
    }

    /// Create from a `BTreeSet<usize>`.
    #[must_use]
    pub fn from_btree_set(set: &BTreeSet<usize>) -> Self {
        let mut result = Self::new();
        for &index in set {
            result.insert(index);
        }
        result
    }

    /// Get raw word access (for advanced use cases like SIMD operations).
    #[inline]
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Get mutable raw word access (for advanced use cases).
    #[inline]
    #[must_use]
    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }
}

impl FromIterator<usize> for BitSet {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut set = Self::new();
        for index in iter {
            set.insert(index);
        }
        set
    }
}

impl<'a> IntoIterator for &'a BitSet {
    type Item = usize;
    type IntoIter = BitSetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over indices in a [`BitSet`].
///
/// Yields indices in ascending order.
pub struct BitSetIter<'a> {
    words: &'a [u64],
    word_idx: usize,
    current_word: u64,
    base_index: usize,
}

impl Iterator for BitSetIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Find next set bit
        while self.current_word == 0 {
            self.word_idx += 1;
            if self.word_idx >= self.words.len() {
                return None;
            }
            self.current_word = self.words[self.word_idx];
            self.base_index = self.word_idx * 64;
        }

        // Extract lowest set bit position
        let bit_pos = self.current_word.trailing_zeros() as usize;
        // Clear lowest set bit
        self.current_word &= self.current_word - 1;
        Some(self.base_index + bit_pos)
    }
}

// Implement BitXorAssign for convenient ^= syntax
impl std::ops::BitXorAssign<&BitSet> for BitSet {
    #[inline]
    fn bitxor_assign(&mut self, rhs: &BitSet) {
        self.symmetric_difference_update(rhs);
    }
}

// Implement BitXor for convenient ^ syntax
impl std::ops::BitXor for &BitSet {
    type Output = BitSet;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.symmetric_difference(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let set = BitSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(!set.contains(0));
    }

    #[test]
    fn test_insert_contains() {
        let mut set = BitSet::new();
        assert!(set.insert(5));
        assert!(!set.insert(5)); // Already present
        assert!(set.contains(5));
        assert!(!set.contains(4));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_remove() {
        let mut set = BitSet::new();
        set.insert(5);
        assert!(set.remove(5));
        assert!(!set.remove(5)); // Already removed
        assert!(!set.contains(5));
        assert!(set.is_empty());
    }

    #[test]
    fn test_large_indices() {
        let mut set = BitSet::new();
        set.insert(0);
        set.insert(63);
        set.insert(64);
        set.insert(1000);

        assert!(set.contains(0));
        assert!(set.contains(63));
        assert!(set.contains(64));
        assert!(set.contains(1000));
        assert!(!set.contains(65));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_xor() {
        let mut a = BitSet::new();
        a.insert(0);
        a.insert(1);

        let mut b = BitSet::new();
        b.insert(1);
        b.insert(2);

        a.symmetric_difference_update(&b);

        // Result should be {0, 2} (1 cancels out)
        assert!(a.contains(0));
        assert!(!a.contains(1));
        assert!(a.contains(2));
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_xor_operator() {
        let mut a = BitSet::new();
        a.insert(0);
        a.insert(1);

        let mut b = BitSet::new();
        b.insert(1);
        b.insert(2);

        a ^= &b;

        assert!(a.contains(0));
        assert!(!a.contains(1));
        assert!(a.contains(2));
    }

    #[test]
    fn test_iter() {
        let mut set = BitSet::new();
        set.insert(0);
        set.insert(5);
        set.insert(64);
        set.insert(100);

        let indices: Vec<_> = set.iter().collect();
        assert_eq!(indices, vec![0, 5, 64, 100]);
    }

    #[test]
    fn test_single() {
        let set = BitSet::single(42);
        assert_eq!(set.len(), 1);
        assert!(set.contains(42));
    }

    #[test]
    fn test_btree_conversion() {
        let btree: BTreeSet<_> = [1, 5, 10, 100].into_iter().collect();
        let bitset = BitSet::from_btree_set(&btree);
        let back = bitset.to_btree_set();
        assert_eq!(btree, back);
    }

    #[test]
    fn test_xor_self_is_empty() {
        let mut set = BitSet::new();
        set.insert(0);
        set.insert(5);

        set.symmetric_difference_update(&set.clone());
        assert!(set.is_empty());
    }

    #[test]
    fn test_from_iterator() {
        let set: BitSet = [1, 5, 10, 100].into_iter().collect();
        assert_eq!(set.len(), 4);
        assert!(set.contains(1));
        assert!(set.contains(5));
        assert!(set.contains(10));
        assert!(set.contains(100));
    }

    #[test]
    fn test_sparse_qec_like_pattern() {
        // Simulate multi-round QEC: measurement at index 950 depends on
        // measurements from rounds 1, 5, and 10 (indices 0, 400, 900)
        let deps: BitSet = [0, 400, 900].into_iter().collect();

        assert_eq!(deps.len(), 3);
        assert!(deps.contains(0));
        assert!(deps.contains(400));
        assert!(deps.contains(900));

        // Check storage efficiency - should have ~15 words (900/64 ≈ 14.06)
        assert_eq!(deps.words().len(), 15); // ceil(901/64) = 15

        // Most words should be zero (sparse)
        let non_zero_words = deps.words().iter().filter(|&&w| w != 0).count();
        assert_eq!(non_zero_words, 3); // Only 3 words have bits set

        // XOR with another sparse set
        let other_deps: BitSet = [0, 500, 900].into_iter().collect();
        let result = &deps ^ &other_deps;

        // {0, 400, 900} XOR {0, 500, 900} = {400, 500}
        assert_eq!(result.len(), 2);
        assert!(!result.contains(0)); // Cancelled
        assert!(result.contains(400));
        assert!(result.contains(500));
        assert!(!result.contains(900)); // Cancelled
    }

    #[test]
    fn test_xor_different_sizes() {
        // Small set XOR'd with large set
        let small: BitSet = [0, 1, 2].into_iter().collect();
        let large: BitSet = [1, 1000].into_iter().collect();

        let result = &small ^ &large;

        // {0, 1, 2} XOR {1, 1000} = {0, 2, 1000}
        assert_eq!(result.len(), 3);
        assert!(result.contains(0));
        assert!(!result.contains(1)); // Cancelled
        assert!(result.contains(2));
        assert!(result.contains(1000));

        // Result should have expanded to fit index 1000
        assert_eq!(result.words().len(), 16); // ceil(1001/64) = 16

        // XOR in the other direction should give same result
        let result2 = &large ^ &small;
        assert_eq!(result, result2);
    }

    #[test]
    fn test_xor_assign_extends() {
        // In-place XOR should extend when needed
        let mut small: BitSet = [0, 1].into_iter().collect();
        let large: BitSet = [1, 5000].into_iter().collect();

        small ^= &large;

        assert_eq!(small.len(), 2);
        assert!(small.contains(0));
        assert!(!small.contains(1)); // Cancelled
        assert!(small.contains(5000));
        assert_eq!(small.words().len(), 79); // ceil(5001/64) = 79
    }
}
