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

use crate::sets::trait_set::Set;
use crate::trait_unsigned_int::UnsignedInt;
use std::fmt::Debug;
use std::ops::{BitAndAssign, BitOrAssign, BitXorAssign};

#[derive(Clone, Debug)]
pub struct VecSetFast<T: UnsignedInt> {
    elements: Vec<T>,
}

#[macro_export]
macro_rules! vecsetfast {
    ($($x:expr),+ $(,)?) => {
        {
            let arr = [$($x),+];
            VecSetFast::from(arr)
        }
    };
}

impl<T: UnsignedInt, const N: usize> From<[T; N]> for VecSetFast<T> {
    fn from(arr: [T; N]) -> Self {
        Self::from_iter(arr)
    }
}

impl<T: UnsignedInt> FromIterator<T> for VecSetFast<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = VecSetFast::new();
        for item in iter {
            set.insert(item);
        }
        set
    }
}

impl<T: UnsignedInt> Default for VecSetFast<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: UnsignedInt> VecSetFast<T> {
    pub fn new() -> Self {
        VecSetFast {
            elements: Vec::new(),
        }
    }

    pub fn new_with_capacity(capacity: usize) -> Self {
        VecSetFast {
            elements: Vec::with_capacity(capacity),
        }
    }

    fn maintain_sorted(&mut self) {
        if self.elements.len() > 1 {
            self.elements.sort_unstable();
            self.elements.dedup();
        }
    }
}

impl<T: UnsignedInt + Debug> VecSetFast<T> {
    pub fn display(&self) {
        println!("{:?}", self.elements);
    }
}

// Implementation for owned iteration (consuming set)
impl<T: UnsignedInt> IntoIterator for VecSetFast<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

// Implementation for borrowed iteration (non-consuming set)
impl<'a, T: UnsignedInt> IntoIterator for &'a VecSetFast<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.iter()
    }
}

#[derive(Clone)]
pub struct Difference<'a, T>
where
    T: UnsignedInt,
{
    iter: std::slice::Iter<'a, T>,
    other: &'a VecSetFast<T>,
}

impl<'a, T: UnsignedInt> Iterator for Difference<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.by_ref().find(|&item| !self.other.contains(item))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.iter.fold(init, |acc, elt| {
            if self.other.contains(elt) {
                acc
            } else {
                f(acc, elt)
            }
        })
    }
}

impl<T> std::fmt::Debug for Difference<'_, T>
where
    T: std::fmt::Debug + UnsignedInt,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<'a, T> Set<'a, T> for VecSetFast<T>
where
    T: UnsignedInt + 'a,
{
    type Iter = std::slice::Iter<'a, T>;
    type Difference = Difference<'a, T>;

    fn new() -> VecSetFast<T> {
        VecSetFast::new()
    }

    fn contains(&self, value: &T) -> bool {
        self.elements.binary_search(value).is_ok()
    }

    fn insert(&mut self, value: T) {
        if let Err(index) = self.elements.binary_search(&value) {
            self.elements.insert(index, value);
        }
    }

    fn remove(&mut self, value: &T) {
        if let Ok(index) = self.elements.binary_search(value) {
            self.elements.remove(index);
        }
    }

    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.elements.retain(f)
    }

    fn union_update(&mut self, rhs: &Self) {
        self.elements.extend_from_slice(&rhs.elements);
        self.maintain_sorted();
    }

    fn symmetric_difference_update(&mut self, rhs: &Self) {
        let mut temp = Self::new();
        for &item in &self.elements {
            if !rhs.contains(&item) {
                temp.insert(item);
            }
        }
        for &item in &rhs.elements {
            if !self.contains(&item) {
                temp.insert(item);
            }
        }
        self.elements = temp.elements;
        self.maintain_sorted();
    }

    fn clear(&mut self) {
        self.elements.clear()
    }

    fn iter(&'a self) -> Self::Iter {
        self.elements.iter()
    }

    fn difference(&'a self, other: &'a Self) -> Self::Difference {
        Difference {
            iter: self.elements.iter(),
            other,
        }
    }
}

impl<T: UnsignedInt> BitXorAssign<&Self> for VecSetFast<T> {
    fn bitxor_assign(&mut self, rhs: &Self) {
        self.symmetric_difference_update(rhs);
    }
}

impl<T: UnsignedInt> BitOrAssign<&Self> for VecSetFast<T> {
    fn bitor_assign(&mut self, rhs: &Self) {
        self.union(rhs);
    }
}

impl<T: UnsignedInt> BitAndAssign<&Self> for VecSetFast<T> {
    fn bitand_assign(&mut self, rhs: &Self) {
        self.intersection(rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let set = VecSetFast::<u32>::new();
        assert!(set.elements.is_empty());
    }

    #[test]
    fn test_new_with_vec() {
        let set: VecSetFast<usize> = vec![4, 5, 6, 4].into_iter().collect();
        assert_eq!(set.elements, vec![4, 5, 6]);
    }

    #[test]
    fn test_new_from() {
        let set = VecSetFast::<u32>::from([4, 5, 6, 4]);
        assert_eq!(set.elements, vec![4, 5, 6]);
    }

    #[test]
    fn test_insert() {
        let mut set = VecSetFast::<u32>::new();
        set.insert(1);
        assert_eq!(set.elements, vec![1]);
        set.insert(5);
        set.insert(1);
        assert_eq!(set.elements, vec![1, 5]);
    }

    #[test]
    fn test_remove() {
        let mut set: VecSetFast<u8> = VecSetFast::from([4, 5, 6, 4]);
        set.remove(&5);
        assert_eq!(set.elements, vec![4, 6]);
        set.remove(&7);
        assert_eq!(set.elements, vec![4, 6]);
    }

    #[test]
    fn test_union() {
        let mut set_a: VecSetFast<u8> = VecSetFast::from([1, 2]);
        let set_b = VecSetFast::from([2, 3]);
        set_a.union(&set_b);
        assert_eq!(set_a.elements, vec![1, 2, 3]);
    }

    #[test]
    fn test_symmetric_difference_update() {
        let mut set_a = VecSetFast::<u32>::from([4, 5, 6, 4]);
        let set_b = VecSetFast::<u32>::from([1, 3, 4]);
        set_a.symmetric_difference_update(&set_b);
        assert_eq!(set_a.elements, vec![1, 3, 5, 6]);
    }

    #[test]
    fn test_bitor_assign() {
        let mut set_a: VecSetFast<u8> = VecSetFast::from([1, 2]);
        let set_b: VecSetFast<u8> = VecSetFast::from([2, 3]);
        set_a |= &set_b;
        assert_eq!(set_a.elements, vec![1, 2, 3])
    }

    #[test]
    fn test_bitxor_assign() {
        let mut set_a: VecSetFast<u8> = VecSetFast::from([1, 2, 3]);
        let set_b: VecSetFast<u8> = VecSetFast::from([2, 3, 4]);
        set_a ^= &set_b;
        assert_eq!(set_a.elements, vec![1, 4])
    }

    #[test]
    fn test_bitand_assign() {
        let mut set_a: VecSetFast<u8> = VecSetFast::from([1, 2, 3]);
        let set_b = VecSetFast::from([2, 3, 4]);
        set_a &= &set_b;
        assert_eq!(set_a.elements, vec![2, 3])
    }

    #[test]
    fn test_difference() {
        let set_a: VecSetFast<u8> = VecSetFast::from([1, 2, 3]);
        let set_b = VecSetFast::from([2, 3, 4]);
        let diff: Vec<_> = set_a.difference(&set_b).cloned().collect();
        assert_eq!(diff, vec![1]);
    }
}
