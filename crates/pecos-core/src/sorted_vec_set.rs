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

//! A sorted vector-based set optimized for merge operations.
//!
//! This module provides [`SortedVecSet`], a set implementation that maintains elements
//! in sorted order, enabling O(n+m) merge-based XOR operations instead of O(n*m).
//!
//! # Performance Characteristics
//!
//! | Operation      | `SortedVecSet` | `VecSet`  | `BitSet` |
//! |----------------|--------------|---------|--------|
//! | contains       | O(log n)     | O(n)    | O(1)   |
//! | insert         | O(n)         | O(n)    | O(1)   |
//! | toggle         | O(n)         | O(n)    | O(1)   |
//! | `xor_assign`     | O(n+m)       | O(n*m)  | O(words) |
//! | iteration      | O(n)         | O(n)    | O(words) |
//!
//! # When to Use
//!
//! - Best for medium-sized sets (16-128 elements) with frequent XOR operations
//! - Maintains sorted iteration order (useful for deterministic output)
//! - Trade-off: O(n) insertion for O(n+m) XOR

use smallvec::SmallVec;

/// Inline capacity for `SortedVecSet` elements.
/// Same as `VecSet` for fair comparison.
const SORTED_VECSET_INLINE_CAPACITY: usize = 8;

/// A stack-optimized buffer for sorted set elements.
type SortedBuffer = SmallVec<[usize; SORTED_VECSET_INLINE_CAPACITY]>;

/// A sorted vector-based set of `usize` indices.
///
/// Elements are always maintained in ascending order, enabling:
/// - O(log n) binary search for contains
/// - O(n+m) merge-based XOR operations
/// - Deterministic iteration order
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SortedVecSet {
    elements: SortedBuffer,
}

impl SortedVecSet {
    /// Create an empty set.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a set with pre-allocated capacity.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            elements: SortedBuffer::with_capacity(capacity),
        }
    }

    /// Returns the number of elements in the set.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Returns true if the set is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Returns the capacity of the set.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.elements.capacity()
    }

    /// Clear all elements from the set.
    #[inline]
    pub fn clear(&mut self) {
        self.elements.clear();
    }

    /// Check if the set contains an element using binary search.
    #[inline]
    #[must_use]
    pub fn contains(&self, value: usize) -> bool {
        self.elements.binary_search(&value).is_ok()
    }

    /// Insert an element, maintaining sorted order.
    /// Returns true if the element was newly inserted.
    #[inline]
    pub fn insert(&mut self, value: usize) -> bool {
        match self.elements.binary_search(&value) {
            Ok(_) => false, // Already present
            Err(pos) => {
                self.elements.insert(pos, value);
                true
            }
        }
    }

    /// Remove an element, maintaining sorted order.
    /// Returns true if the element was present.
    #[inline]
    pub fn remove(&mut self, value: usize) -> bool {
        match self.elements.binary_search(&value) {
            Ok(pos) => {
                self.elements.remove(pos);
                true
            }
            Err(_) => false,
        }
    }

    /// Toggle an element (insert if absent, remove if present).
    #[inline]
    pub fn toggle(&mut self, value: usize) {
        match self.elements.binary_search(&value) {
            Ok(pos) => {
                self.elements.remove(pos);
            }
            Err(pos) => {
                self.elements.insert(pos, value);
            }
        }
    }

    /// XOR (symmetric difference) with another set using merge algorithm.
    /// This is O(n+m) instead of O(n*m).
    #[inline]
    pub fn xor_assign(&mut self, other: &Self) {
        // Fast path for empty other
        if other.is_empty() {
            return;
        }

        // Fast path for empty self
        if self.is_empty() {
            self.elements.clone_from(&other.elements);
            return;
        }

        // Merge-based XOR
        let result = Self::merge_xor(&self.elements, &other.elements);
        self.elements = result;
    }

    /// Merge-based XOR for sorted arrays - O(n+m)
    #[inline]
    fn merge_xor(a: &[usize], b: &[usize]) -> SortedBuffer {
        let mut result = SortedBuffer::with_capacity(a.len() + b.len());
        let mut i = 0;
        let mut j = 0;

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
                    // Element in both - XOR cancels, skip both
                    i += 1;
                    j += 1;
                }
            }
        }

        // Add remaining elements
        result.extend_from_slice(&a[i..]);
        result.extend_from_slice(&b[j..]);

        result
    }

    /// Iterate over elements in sorted order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.elements.iter().copied()
    }

    /// Get the elements as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        &self.elements
    }

    /// Take the contents, leaving self empty but with capacity preserved.
    #[inline]
    #[must_use]
    pub fn take_clearing(&mut self) -> Self {
        let taken: SortedBuffer = self.elements.drain(..).collect();
        Self { elements: taken }
    }

    /// Clear the set and insert a single element.
    #[inline]
    pub fn set_single(&mut self, value: usize) {
        self.elements.clear();
        self.elements.push(value);
    }

    /// Count elements in the intersection using merge algorithm.
    #[inline]
    #[must_use]
    pub fn intersection_count(&self, other: &Self) -> usize {
        let mut count = 0;
        let mut i = 0;
        let mut j = 0;

        while i < self.elements.len() && j < other.elements.len() {
            match self.elements[i].cmp(&other.elements[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    count += 1;
                    i += 1;
                    j += 1;
                }
            }
        }

        count
    }

    /// XOR elements in the intersection into target using merge algorithm.
    #[inline]
    pub fn xor_intersection_into(&self, other: &Self, target: &mut Self) {
        let mut i = 0;
        let mut j = 0;

        while i < self.elements.len() && j < other.elements.len() {
            match self.elements[i].cmp(&other.elements[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    target.toggle(self.elements[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
    }

    /// XOR elements in the symmetric difference into target.
    #[inline]
    pub fn xor_symmetric_difference_into(&self, other: &Self, target: &mut Self) {
        let mut i = 0;
        let mut j = 0;

        while i < self.elements.len() && j < other.elements.len() {
            match self.elements[i].cmp(&other.elements[j]) {
                std::cmp::Ordering::Less => {
                    target.toggle(self.elements[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    target.toggle(other.elements[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    // In both sets - not in symmetric difference
                    i += 1;
                    j += 1;
                }
            }
        }

        // Remaining elements from self
        while i < self.elements.len() {
            target.toggle(self.elements[i]);
            i += 1;
        }

        // Remaining elements from other
        while j < other.elements.len() {
            target.toggle(other.elements[j]);
            j += 1;
        }
    }
}

impl FromIterator<usize> for SortedVecSet {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut set = Self::new();
        for item in iter {
            set.insert(item);
        }
        set
    }
}

// Implement IndexSet trait
impl crate::index_set::IndexSet for SortedVecSet {
    type Iter<'a> = std::iter::Copied<std::slice::Iter<'a, usize>>;

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
    fn xor_assign(&mut self, other: &Self) {
        Self::xor_assign(self, other);
    }

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        self.elements.iter().copied()
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
    fn set_single(&mut self, index: usize) {
        Self::set_single(self, index);
    }

    #[inline]
    fn intersection_count(&self, other: &Self) -> usize {
        Self::intersection_count(self, other)
    }

    #[inline]
    fn xor_intersection_into(&self, other: &Self, target: &mut Self) {
        Self::xor_intersection_into(self, other, target);
    }

    #[inline]
    fn xor_symmetric_difference_into(&self, other: &Self, target: &mut Self) {
        Self::xor_symmetric_difference_into(self, other, target);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_empty() {
        let set = SortedVecSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn test_insert_maintains_order() {
        let mut set = SortedVecSet::new();
        set.insert(5);
        set.insert(1);
        set.insert(3);
        set.insert(2);
        set.insert(4);

        assert_eq!(set.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_insert_no_duplicates() {
        let mut set = SortedVecSet::new();
        assert!(set.insert(1));
        assert!(!set.insert(1));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_contains() {
        let mut set = SortedVecSet::new();
        set.insert(1);
        set.insert(5);
        set.insert(10);

        assert!(set.contains(1));
        assert!(set.contains(5));
        assert!(set.contains(10));
        assert!(!set.contains(0));
        assert!(!set.contains(3));
        assert!(!set.contains(100));
    }

    #[test]
    fn test_remove() {
        let mut set = SortedVecSet::new();
        set.insert(1);
        set.insert(2);
        set.insert(3);

        assert!(set.remove(2));
        assert!(!set.remove(2));
        assert_eq!(set.as_slice(), &[1, 3]);
    }

    #[test]
    fn test_toggle() {
        let mut set = SortedVecSet::new();
        set.toggle(1);
        assert!(set.contains(1));

        set.toggle(1);
        assert!(!set.contains(1));

        set.toggle(3);
        set.toggle(1);
        set.toggle(2);
        assert_eq!(set.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_xor_assign() {
        let mut a: SortedVecSet = [1, 2, 3, 4].into_iter().collect();
        let b: SortedVecSet = [3, 4, 5, 6].into_iter().collect();

        a.xor_assign(&b);
        assert_eq!(a.as_slice(), &[1, 2, 5, 6]);
    }

    #[test]
    fn test_xor_assign_empty() {
        let mut a: SortedVecSet = [1, 2, 3].into_iter().collect();
        let b = SortedVecSet::new();

        a.xor_assign(&b);
        assert_eq!(a.as_slice(), &[1, 2, 3]);

        let mut c = SortedVecSet::new();
        let d: SortedVecSet = [1, 2, 3].into_iter().collect();
        c.xor_assign(&d);
        assert_eq!(c.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_intersection_count() {
        let a: SortedVecSet = [1, 2, 3, 4, 5].into_iter().collect();
        let b: SortedVecSet = [3, 4, 5, 6, 7].into_iter().collect();

        assert_eq!(a.intersection_count(&b), 3);
        assert_eq!(b.intersection_count(&a), 3);
    }

    #[test]
    fn test_set_single() {
        let mut set: SortedVecSet = [1, 2, 3, 4, 5].into_iter().collect();
        set.set_single(10);
        assert_eq!(set.as_slice(), &[10]);
    }

    #[test]
    fn test_take_clearing() {
        let mut set: SortedVecSet = [1, 2, 3].into_iter().collect();
        let taken = set.take_clearing();

        assert!(set.is_empty());
        assert_eq!(taken.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_from_iter() {
        let set: SortedVecSet = [5, 1, 3, 2, 4].into_iter().collect();
        assert_eq!(set.as_slice(), &[1, 2, 3, 4, 5]);
    }
}
