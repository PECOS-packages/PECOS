// Copyright 2024 The PECOS Developers
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

mod iterators;
mod operators;
mod set_impl;

use crate::{Element, Set};
use core::slice::Iter;
use smallvec::SmallVec;

/// Inline capacity for `VecSet` elements.
/// Stabilizer weights are typically 2-4 elements in surface codes.
/// Using 8 provides good performance for small circuits while avoiding
/// significant memory overhead at larger scales.
const VECSET_INLINE_CAPACITY: usize = 8;

/// A stack-optimized small buffer for set elements.
/// Uses inline storage for up to 8 elements before spilling to heap.
pub type SetBuffer<E> = SmallVec<[E; VECSET_INLINE_CAPACITY]>;

#[derive(PartialEq, Clone, Debug)]
pub struct VecSet<E: Element> {
    pub(crate) elements: SetBuffer<E>,
}

#[macro_export]
macro_rules! vecset {
    ($($x:expr),+ $(,)?) => {
        {
            let arr = [$($x),+];
            VecSet::from(arr)
        }
    };
}

impl<E: Element, const N: usize> From<[E; N]> for VecSet<E> {
    #[inline]
    fn from(arr: [E; N]) -> Self {
        Self::from_iter(arr)
    }
}

impl<E: Element> FromIterator<E> for VecSet<E> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = E>>(iter: I) -> Self {
        let mut set = Self::new();
        for item in iter {
            set.insert(item);
        }
        set
    }
}

impl<E: Element> VecSet<E> {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, E> {
        self.elements.iter()
    }

    /// Get the elements as a slice.
    ///
    /// This provides direct access to the underlying storage for optimized
    /// operations that can work with slices instead of iterators.
    #[inline]
    pub fn as_slice(&self) -> &[E] {
        &self.elements
    }

    #[inline]
    #[must_use]
    pub fn elements(&self) -> &[E] {
        &self.elements
    }

    /// Clear the set and set it to contain exactly one element.
    /// This is an optimized operation for reset scenarios where we know
    /// the set should contain exactly one element.
    ///
    /// Unlike `clear()` + `insert()`, this skips the `contains()` check
    /// since we know the set is empty after clearing.
    #[inline]
    pub fn set_single(&mut self, value: E) {
        self.elements.clear();
        self.elements.push(value);
    }

    /// Fused operation: XOR each element in `self ∩ other` into `target`.
    /// Equivalent to: `for i in self.intersection(other) { target ^= i; }`
    /// but avoids iterator creation overhead and inlines the XOR operation.
    #[inline]
    pub fn xor_intersection_into(&self, other: &Self, target: &mut Self) {
        // Iterate over the smaller set for better performance
        let (smaller, larger) = if self.elements.len() <= other.elements.len() {
            (&self.elements, &other.elements)
        } else {
            (&other.elements, &self.elements)
        };
        for &elt in smaller {
            if larger.contains(&elt) {
                // Inline symmetric_difference_item_update
                if let Some(pos) = target.elements.iter().position(|x| *x == elt) {
                    target.elements.swap_remove(pos);
                } else {
                    target.elements.push(elt);
                }
            }
        }
    }

    /// Fused operation: XOR each element in `self ⊕ other` (symmetric difference) into `target`.
    /// Equivalent to: `for i in self.symmetric_difference(other) { target ^= i; }`
    /// but avoids iterator creation overhead and inlines the XOR operation.
    #[inline]
    pub fn xor_symmetric_difference_into(&self, other: &Self, target: &mut Self) {
        // Process elements in self that are not in other
        for &elt in &self.elements {
            if !other.elements.contains(&elt) {
                if let Some(pos) = target.elements.iter().position(|x| *x == elt) {
                    target.elements.swap_remove(pos);
                } else {
                    target.elements.push(elt);
                }
            }
        }
        // Process elements in other that are not in self
        for &elt in &other.elements {
            if !self.elements.contains(&elt) {
                if let Some(pos) = target.elements.iter().position(|x| *x == elt) {
                    target.elements.swap_remove(pos);
                } else {
                    target.elements.push(elt);
                }
            }
        }
    }

    /// Take the contents of this set, leaving it empty but with capacity preserved.
    ///
    /// Unlike `std::mem::take`, this preserves the allocated capacity of the source
    /// set, which is important for performance when the set will be reused.
    #[inline]
    #[must_use]
    pub fn take_clearing(&mut self) -> Self {
        // Drain elements into a new set, leaving self empty with capacity
        let taken_elements: SetBuffer<E> = self.elements.drain(..).collect();
        Self {
            elements: taken_elements,
        }
    }

    /// Count elements in the intersection of `self` and `other`.
    /// Equivalent to: `self.intersection(other).count()`
    /// but avoids iterator struct creation overhead.
    #[inline]
    pub fn intersection_count(&self, other: &Self) -> usize {
        // Iterate over smaller set for better performance
        let (smaller, larger) = if self.elements.len() <= other.elements.len() {
            (&self.elements, &other.elements)
        } else {
            (&other.elements, &self.elements)
        };
        smaller.iter().filter(|x| larger.contains(x)).count()
    }
}

// Cross-type methods for VecSet<usize> to work with BitSet
impl VecSet<usize> {
    /// XOR elements in the intersection of `self` and `other` into a `BitSet` target.
    ///
    /// This is useful for hybrid implementations where Pauli data uses `VecSet`
    /// but sign data uses `BitSet`.
    #[inline]
    pub fn xor_intersection_into_bitset(&self, other: &Self, target: &mut crate::BitSet) {
        // Iterate over the smaller set for better performance
        let (smaller, larger) = if self.elements.len() <= other.elements.len() {
            (&self.elements, &other.elements)
        } else {
            (&other.elements, &self.elements)
        };
        for &elt in smaller {
            if larger.contains(&elt) {
                target.toggle(elt);
            }
        }
    }

    /// XOR elements in the symmetric difference of `self` and `other` into a `BitSet` target.
    ///
    /// This is useful for hybrid implementations where Pauli data uses `VecSet`
    /// but sign data uses `BitSet`.
    #[inline]
    pub fn xor_symmetric_difference_into_bitset(&self, other: &Self, target: &mut crate::BitSet) {
        // Elements in self but not in other
        for &elt in &self.elements {
            if !other.elements.contains(&elt) {
                target.toggle(elt);
            }
        }
        // Elements in other but not in self
        for &elt in &other.elements {
            if !self.elements.contains(&elt) {
                target.toggle(elt);
            }
        }
    }
}

impl<E: Element> Default for VecSet<E> {
    #[inline]
    fn default() -> Self {
        Self {
            elements: SetBuffer::new(),
        }
    }
}

impl<'a, E: Element> IntoIterator for &'a VecSet<E> {
    type Item = &'a E;
    type IntoIter = Iter<'a, E>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Implement IndexSet for VecSet<usize>
impl crate::index_set::IndexSet for VecSet<usize> {
    type Iter<'a> = std::iter::Copied<std::slice::Iter<'a, usize>>;

    #[inline]
    fn new() -> Self {
        Self::new()
    }

    #[inline]
    fn insert(&mut self, index: usize) -> bool {
        use crate::sets::set::Set;
        if Set::contains(self, &index) {
            false
        } else {
            Set::insert(self, index);
            true
        }
    }

    #[inline]
    fn remove(&mut self, index: usize) -> bool {
        use crate::sets::set::Set;
        if Set::contains(self, &index) {
            Set::remove(self, &index);
            true
        } else {
            false
        }
    }

    #[inline]
    fn contains(&self, index: usize) -> bool {
        use crate::sets::set::Set;
        Set::contains(self, &index)
    }

    #[inline]
    fn toggle(&mut self, index: usize) {
        use crate::sets::set::Set;
        Set::symmetric_difference_item_update(self, &index);
    }

    #[inline]
    fn xor_assign(&mut self, other: &Self) {
        use crate::sets::set::Set;
        Set::symmetric_difference_update(self, other);
    }

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        self.elements.iter().copied()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        use crate::sets::set::Set;
        Set::is_empty(self)
    }

    #[inline]
    fn len(&self) -> usize {
        use crate::sets::set::Set;
        Set::len(self)
    }

    #[inline]
    fn clear(&mut self) {
        use crate::sets::set::Set;
        Set::clear(self);
    }

    #[inline]
    fn set_single(&mut self, index: usize) {
        // Use optimized version that skips contains check
        self.elements.clear();
        self.elements.push(index);
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

    #[inline]
    fn take_clearing(&mut self) -> Self {
        Self::take_clearing(self)
    }
}

#[cfg(test)]
mod tests {
    use super::super::set::Set;
    use super::VecSet;

    #[test]
    fn test_new() {
        let set = VecSet::<u32>::new();
        assert!(set.elements.is_empty());
    }

    #[test]
    fn test_with_capacity() {
        let set = VecSet::<u32>::with_capacity(10);
        assert!(set.elements.is_empty());
        // SmallVec inline capacity is 8, so requesting 10 should give at least 10
        assert!(set.capacity() >= 10);
    }

    #[test]
    fn test_new_with_vec() {
        let set: VecSet<usize> = vec![4, 5, 6, 4].into_iter().collect();
        assert_eq!(set.elements.as_slice(), &[4, 5, 6]);
    }

    #[test]
    fn test_new_from() {
        let set = VecSet::<u32>::from([4, 5, 6, 4]);
        assert_eq!(set.elements.as_slice(), &[4, 5, 6]);
    }

    #[test]
    fn test_insert() {
        let mut set = VecSet::<u32>::new();
        set.insert(1);
        assert_eq!(set.elements.as_slice(), &[1]);
        set.insert(5);
        set.insert(1);
        assert_eq!(set.elements.as_slice(), &[1, 5]);
    }

    #[test]
    fn test_remove() {
        let mut set: VecSet<u8> = VecSet::from([4, 5, 6, 4]);
        set.remove(&5);
        assert_eq!(set.elements.as_slice(), &[4, 6]);
        set.remove(&7);
        assert_eq!(set.elements.as_slice(), &[4, 6]);
    }

    #[test]
    fn test_union() {
        let mut set_a: VecSet<u8> = VecSet::from([1, 2]);
        let set_b = VecSet::from([2, 3]);
        set_a.union_update(&set_b);
        assert_eq!(set_a.elements.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_symmetric_difference_update() {
        let mut set_a = VecSet::<u32>::from([4, 5, 6, 4]);
        let set_b = VecSet::<u32>::from([1, 3, 4]);
        set_a.symmetric_difference_update(&set_b);
        // Use sorted comparison since swap_remove doesn't preserve order
        let mut result: Vec<_> = set_a.elements.to_vec();
        result.sort_unstable();
        assert_eq!(result, vec![1, 3, 5, 6]);
    }

    #[test]
    fn test_intersection_update() {
        let mut set_a: VecSet<u8> = VecSet::from([1, 2, 3]);
        let set_b = VecSet::from([2, 3, 4]);
        set_a.intersection_update(&set_b);
        assert_eq!(set_a.elements.as_slice(), &[2, 3]);
    }

    #[test]
    fn test_intersection() {
        let set_a: VecSet<u8> = VecSet::from([1, 2, 3]);
        let set_b = VecSet::from([2, 3, 4]);
        let intersection: Vec<_> = set_a.intersection(&set_b).copied().collect();
        assert_eq!(intersection, vec![2, 3]);
    }

    #[test]
    fn test_symmetric_difference() {
        let set_a = VecSet::<u32>::from([4, 5, 6, 4]);
        let set_b = VecSet::<u32>::from([1, 3, 4]);
        let sym_diff: Vec<_> = set_a.symmetric_difference(&set_b).copied().collect();
        assert_eq!(sym_diff, vec![5, 6, 1, 3]);
    }

    #[test]
    fn test_bitor_assign() {
        let mut set_a: VecSet<u8> = VecSet::from([1, 2]);
        let set_b: VecSet<u8> = VecSet::from([2, 3]);
        set_a |= &set_b;
        assert_eq!(set_a.elements.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_bitxor_assign() {
        let mut set_a: VecSet<u8> = VecSet::from([1, 2, 3]);
        let set_b: VecSet<u8> = VecSet::from([2, 3, 4]);
        set_a ^= &set_b;
        assert_eq!(set_a.elements.as_slice(), &[1, 4]);
    }

    #[test]
    fn test_bitand_assign() {
        let mut set_a: VecSet<u8> = VecSet::from([1, 2, 3]);
        let set_b = VecSet::from([2, 3, 4]);
        set_a &= &set_b;
        assert_eq!(set_a.elements.as_slice(), &[2, 3]);
    }

    #[test]
    fn test_bitand_assign_single_element_ref() {
        let mut set_a: VecSet<u8> = VecSet::from([1, 2, 3]);
        let element: u8 = 2;
        set_a &= &element;
        assert_eq!(set_a.elements.as_slice(), &[2]);

        let mut set_b: VecSet<u8> = VecSet::from([1, 2, 3]);
        let non_existing_element: u8 = 4;
        set_b &= &non_existing_element;
        assert!(set_b.elements.is_empty());
    }

    #[test]
    fn test_difference() {
        let set_a: VecSet<u8> = VecSet::from([1, 2, 3]);
        let set_b = VecSet::from([2, 3, 4]);
        let diff: Vec<_> = set_a.difference(&set_b).copied().collect();
        assert_eq!(diff, vec![1]);
    }

    #[test]
    fn test_capacity() {
        let mut set = VecSet::<u32>::with_capacity(10);
        assert!(set.capacity() >= 10);
        set.insert(1);
        assert!(set.capacity() >= 1);
    }

    #[test]
    fn test_clear() {
        let mut set = VecSet::<u32>::from([1, 2, 3]);
        assert!(!set.is_empty());
        set.clear();
        assert!(set.is_empty());
    }

    #[test]
    fn test_symmetric_difference_item_update() {
        let mut set = VecSet::<u32>::from([1, 2, 3]);
        set.symmetric_difference_item_update(&2);
        assert_eq!(set.elements.as_slice(), &[1, 3]);
        set.symmetric_difference_item_update(&4);
        assert_eq!(set.elements.as_slice(), &[1, 3, 4]);
    }

    #[test]
    fn test_union_item_update() {
        let mut set = VecSet::<u32>::from([1, 2, 3]);
        set.union_item_update(&4);
        assert_eq!(set.elements.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_sub_ref() {
        let set_a: VecSet<u8> = VecSet::from([1, 2, 3]);
        let set_b: VecSet<u8> = VecSet::from([2, 3, 4]);
        let difference: Vec<_> = set_a.difference_ref(&set_b).copied().collect();
        assert_eq!(difference, vec![1]);
    }

    #[test]
    fn test_a_xor_b_sub_c() {
        let mut set_a: VecSet<u8> = VecSet::from([1, 2, 3, 4, 5]);
        let set_b: VecSet<u8> = VecSet::from([2, 3, 6]);
        let set_c: VecSet<u8> = VecSet::from([3]);
        set_a ^= set_b.difference(&set_c).copied().collect::<VecSet<_>>();
        // Use sorted comparison since swap_remove doesn't preserve order
        let mut result: Vec<_> = set_a.elements.to_vec();
        result.sort_unstable();
        assert_eq!(result, vec![1, 3, 4, 5, 6]);
    }

    #[test]
    fn test_a_xor_b_sub_c_glyphs() {
        let mut set_a: VecSet<u8> = VecSet::from([1, 2, 3, 4, 5]);
        let set_b: VecSet<u8> = VecSet::from([2, 3, 6]);
        let set_c: VecSet<u8> = VecSet::from([3]);
        set_a ^= &set_b - &set_c; // TODO: Get this to work for Set
        // Use sorted comparison since swap_remove doesn't preserve order
        let mut result: Vec<_> = set_a.elements.to_vec();
        result.sort_unstable();
        assert_eq!(result, vec![1, 3, 4, 5, 6]);
    }

    #[test]
    fn test_take_clearing() {
        let mut set = VecSet::<u32>::from([1, 2, 3, 4, 5]);
        let original_capacity = set.capacity();

        let taken = set.take_clearing();

        // Taken set should have the original elements
        assert_eq!(taken.elements.as_slice(), &[1, 2, 3, 4, 5]);

        // Original set should be empty
        assert!(set.is_empty());

        // Original set should preserve capacity (for heap-allocated sets)
        // For SmallVec with inline storage, capacity is always at least VECSET_INLINE_CAPACITY
        assert!(set.capacity() >= original_capacity.min(8));
    }
}
