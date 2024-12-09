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

#[derive(Clone, Debug)]
pub struct VecSet<E: Element> {
    elements: Vec<E>,
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
}

impl<E: Element> Default for VecSet<E> {
    #[inline]
    fn default() -> Self {
        Self {
            elements: Vec::default(),
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
        assert_eq!(set.capacity(), 10);
    }

    #[test]
    fn test_new_with_vec() {
        let set: VecSet<usize> = vec![4, 5, 6, 4].into_iter().collect();
        assert_eq!(set.elements, vec![4, 5, 6]);
    }

    #[test]
    fn test_new_from() {
        let set = VecSet::<u32>::from([4, 5, 6, 4]);
        assert_eq!(set.elements, vec![4, 5, 6]);
    }

    #[test]
    fn test_insert() {
        let mut set = VecSet::<u32>::new();
        set.insert(1);
        assert_eq!(set.elements, vec![1]);
        set.insert(5);
        set.insert(1);
        assert_eq!(set.elements, vec![1, 5]);
    }

    #[test]
    fn test_remove() {
        let mut set: VecSet<u8> = VecSet::from([4, 5, 6, 4]);
        set.remove(&5);
        assert_eq!(set.elements, vec![4, 6]);
        set.remove(&7);
        assert_eq!(set.elements, vec![4, 6]);
    }

    #[test]
    fn test_union() {
        let mut set_a: VecSet<u8> = VecSet::from([1, 2]);
        let set_b = VecSet::from([2, 3]);
        set_a.union_update(&set_b);
        assert_eq!(set_a.elements, vec![1, 2, 3]);
    }

    #[test]
    fn test_symmetric_difference_update() {
        let mut set_a = VecSet::<u32>::from([4, 5, 6, 4]);
        let set_b = VecSet::<u32>::from([1, 3, 4]);
        set_a.symmetric_difference_update(&set_b);
        assert_eq!(set_a.elements, vec![5, 6, 1, 3]);
    }

    #[test]
    fn test_intersection_update() {
        let mut set_a: VecSet<u8> = VecSet::from([1, 2, 3]);
        let set_b = VecSet::from([2, 3, 4]);
        set_a.intersection_update(&set_b);
        assert_eq!(set_a.elements, vec![2, 3]);
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
        assert_eq!(set_a.elements, vec![1, 2, 3]);
    }

    #[test]
    fn test_bitxor_assign() {
        let mut set_a: VecSet<u8> = VecSet::from([1, 2, 3]);
        let set_b: VecSet<u8> = VecSet::from([2, 3, 4]);
        set_a ^= &set_b;
        assert_eq!(set_a.elements, vec![1, 4]);
    }

    #[test]
    fn test_bitand_assign() {
        let mut set_a: VecSet<u8> = VecSet::from([1, 2, 3]);
        let set_b = VecSet::from([2, 3, 4]);
        set_a &= &set_b;
        assert_eq!(set_a.elements, vec![2, 3]);
    }

    #[test]
    fn test_bitand_assign_single_element_ref() {
        let mut set_a: VecSet<u8> = VecSet::from([1, 2, 3]);
        let element: u8 = 2;
        set_a &= &element;
        assert_eq!(set_a.elements, vec![2]);

        let mut set_b: VecSet<u8> = VecSet::from([1, 2, 3]);
        let non_existing_element: u8 = 4;
        set_b &= &non_existing_element;
        assert_eq!(set_b.elements, Vec::<u8>::new());
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
        assert_eq!(set.capacity(), 10);
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
        assert_eq!(set.elements, vec![1, 3]);
        set.symmetric_difference_item_update(&4);
        assert_eq!(set.elements, vec![1, 3, 4]);
    }

    #[test]
    fn test_union_item_update() {
        let mut set = VecSet::<u32>::from([1, 2, 3]);
        set.union_item_update(&4);
        assert_eq!(set.elements, vec![1, 2, 3, 4]);
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
        assert_eq!(set_a.elements, vec![1, 3, 4, 5, 6]);
    }

    #[test]
    fn test_a_xor_b_sub_c_glyphs() {
        let mut set_a: VecSet<u8> = VecSet::from([1, 2, 3, 4, 5]);
        let set_b: VecSet<u8> = VecSet::from([2, 3, 6]);
        let set_c: VecSet<u8> = VecSet::from([3]);
        set_a ^= &set_b - &set_c; // TODO: Get this to work for Set
        assert_eq!(set_a.elements, vec![1, 3, 4, 5, 6]);
    }
}
