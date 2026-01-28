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

    /// Toggle an index in the set (insert if absent, remove if present).
    ///
    /// This is equivalent to XOR with a single-element set but without
    /// creating a temporary `BitSet`.
    ///
    /// Returns `true` if the index is now present, `false` if it was removed.
    #[inline]
    pub fn toggle(&mut self, index: usize) -> bool {
        let word_idx = index / 64;
        let bit_idx = index % 64;
        let mask = 1u64 << bit_idx;

        // Extend if necessary
        if word_idx >= self.words.len() {
            self.words.resize(word_idx + 1, 0);
        }

        self.words[word_idx] ^= mask;
        (self.words[word_idx] & mask) != 0
    }

    /// Toggle an index without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure the `BitSet` was created with sufficient capacity
    /// via `with_capacity(max_index)` where `max_index > index`.
    ///
    /// This is an optimization for hot paths like CX gate implementation where
    /// we know all `BitSets` are pre-sized to `num_qubits`.
    #[inline]
    pub fn toggle_unchecked(&mut self, index: usize) {
        let word_idx = index / 64;
        let bit_idx = index % 64;
        // SAFETY: Caller guarantees capacity is sufficient
        // Use get_unchecked_mut for maximum performance
        unsafe {
            *self.words.get_unchecked_mut(word_idx) ^= 1u64 << bit_idx;
        }
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

    /// Take the contents of this set, leaving it empty but with capacity preserved.
    ///
    /// Unlike `std::mem::take`, this preserves the allocated capacity of the source
    /// set, which is important for performance when the set will be reused.
    ///
    /// This is used in measurement operations where we take a row's contents
    /// but the row will be populated again in subsequent operations.
    #[inline]
    #[must_use]
    pub fn take_clearing(&mut self) -> Self {
        // Swap words with an empty Vec, leaving capacity in self
        let taken_words = std::mem::take(&mut self.words);
        // Restore capacity by creating a zeroed Vec of the same length
        self.words = vec![0u64; taken_words.len()];
        Self { words: taken_words }
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

    /// XOR (toggle) each index from an iterator into this set.
    ///
    /// This is useful for cross-type operations where the source is not a `BitSet`
    /// (e.g., iterating over a `VecSet` and `XORing` into a `BitSet`).
    #[inline]
    pub fn xor_assign_iter(&mut self, iter: impl Iterator<Item = usize>) {
        for index in iter {
            self.toggle(index);
        }
    }

    /// XOR (toggle) each index from a slice into this set.
    ///
    /// Optimized version with inlined toggle logic.
    #[inline]
    pub fn xor_assign_slice(&mut self, indices: &[usize]) {
        for &index in indices {
            let word_idx = index / 64;
            let bit_idx = index % 64;

            if word_idx >= self.words.len() {
                self.words.resize(word_idx + 1, 0);
            }
            self.words[word_idx] ^= 1u64 << bit_idx;
        }
    }

    /// XOR indices that are present in both `iter` and `other` into this set.
    ///
    /// This is useful for cross-type sign propagation where `other` is a different set type.
    /// For each index in `iter`, if it's also in `other`, toggle it in `self`.
    #[inline]
    pub fn xor_intersection_iter(&mut self, iter: impl Iterator<Item = usize>, other: &Self) {
        for index in iter {
            if other.contains(index) {
                self.toggle(index);
            }
        }
    }

    /// XOR indices from a slice that are present in `other` into this set.
    ///
    /// Optimized version with inlined contains and toggle.
    #[inline]
    pub fn xor_intersection_slice(&mut self, indices: &[usize], other: &Self) {
        for &index in indices {
            let word_idx = index / 64;
            let bit_idx = index % 64;
            let mask = 1u64 << bit_idx;

            // Check if index is in other (inlined contains)
            let in_other = word_idx < other.words.len() && (other.words[word_idx] & mask) != 0;

            if in_other {
                // Toggle in self (inlined toggle)
                if word_idx >= self.words.len() {
                    self.words.resize(word_idx + 1, 0);
                }
                self.words[word_idx] ^= mask;
            }
        }
    }

    /// Count elements present in both `iter` and this set.
    ///
    /// This is useful for cross-type intersection counting.
    #[inline]
    pub fn intersection_count_iter(&self, iter: impl Iterator<Item = usize>) -> usize {
        iter.filter(|&index| self.contains(index)).count()
    }

    /// Count elements from a slice that are present in this set.
    ///
    /// Optimized version with inlined contains check.
    #[inline]
    #[must_use]
    pub fn intersection_count_slice(&self, indices: &[usize]) -> usize {
        let mut count = 0;
        for &index in indices {
            let word_idx = index / 64;
            let bit_idx = index % 64;
            if word_idx < self.words.len()
                && (unsafe { *self.words.get_unchecked(word_idx) } & (1u64 << bit_idx)) != 0
            {
                count += 1;
            }
        }
        count
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

// Implement BitXorAssign for single element (toggle)
impl std::ops::BitXorAssign<&usize> for BitSet {
    #[inline]
    fn bitxor_assign(&mut self, rhs: &usize) {
        self.toggle(*rhs);
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

// Implement IndexSet for BitSet
impl crate::index_set::IndexSet for BitSet {
    type Iter<'a> = BitSetIter<'a>;

    #[inline]
    fn new() -> Self {
        Self::new()
    }

    #[inline]
    fn with_capacity(max_index: usize) -> Self {
        Self::with_capacity(max_index)
    }

    #[inline]
    fn insert(&mut self, index: usize) -> bool {
        Self::insert(self, index)
    }

    #[inline]
    fn remove(&mut self, index: usize) -> bool {
        Self::remove(self, index)
    }

    #[inline]
    fn contains(&self, index: usize) -> bool {
        Self::contains(self, index)
    }

    #[inline]
    fn toggle(&mut self, index: usize) {
        Self::toggle(self, index);
    }

    #[inline]
    fn toggle_unchecked(&mut self, index: usize) {
        Self::toggle_unchecked(self, index);
    }

    #[inline]
    fn xor_assign(&mut self, other: &Self) {
        self.symmetric_difference_update(other);
    }

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        Self::iter(self)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        Self::is_empty(self)
    }

    #[inline]
    fn len(&self) -> usize {
        Self::len(self)
    }

    #[inline]
    fn clear(&mut self) {
        Self::clear(self);
    }

    #[inline]
    fn take_clearing(&mut self) -> Self {
        Self::take_clearing(self)
    }

    #[inline]
    fn intersection_count(&self, other: &Self) -> usize {
        let min_len = self.words.len().min(other.words.len());
        let mut count = 0;
        for i in 0..min_len {
            count += (self.words[i] & other.words[i]).count_ones() as usize;
        }
        count
    }

    #[inline]
    fn xor_intersection_into(&self, other: &Self, target: &mut Self) {
        let min_len = self.words.len().min(other.words.len());
        // Ensure target has enough words
        if target.words.len() < min_len {
            target.words.resize(min_len, 0);
        }
        for i in 0..min_len {
            let intersection = self.words[i] & other.words[i];
            if i < target.words.len() {
                target.words[i] ^= intersection;
            }
        }
    }

    #[inline]
    fn xor_symmetric_difference_into(&self, other: &Self, target: &mut Self) {
        let max_len = self.words.len().max(other.words.len());
        // Ensure target has enough words
        if target.words.len() < max_len {
            target.words.resize(max_len, 0);
        }
        // XOR elements that are in self XOR other into target
        for i in 0..max_len {
            let self_word = if i < self.words.len() {
                self.words[i]
            } else {
                0
            };
            let other_word = if i < other.words.len() {
                other.words[i]
            } else {
                0
            };
            let symmetric_diff = self_word ^ other_word;
            target.words[i] ^= symmetric_diff;
        }
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
