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
use crate::sets::vecset::VecSet;
use std::ops::{BitAndAssign, BitOrAssign, BitXorAssign};

#[derive(Clone, Debug)]
pub struct UVecSet {
    elements: Vec<usize>,
}

impl UVecSet {
    fn display(&self) {
        println!("{:?}", self.elements);
    }
}

impl FromIterator<usize> for UVecSet {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        let mut set = UVecSet::new();
        for item in iter {
            set.insert(item);
        }
        set
    }
}

// Implementation for owned iteration (consuming set)
impl IntoIterator for UVecSet {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

// Implementation for borrowed iteration (non-consuming set)
impl<'a> IntoIterator for &'a UVecSet {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.iter()
    }
}

impl<const N: usize> From<[usize; N]> for UVecSet {
    fn from(arr: [usize; N]) -> Self {
        Self::from_iter(arr.iter().cloned())
    }
}

impl Set<usize> for UVecSet {
    fn new() -> UVecSet {
        UVecSet {
            elements: Vec::new(),
        }
    }

    fn contains(&self, value: &usize) -> bool {
        self.elements.contains(value)
    }

    fn insert(&mut self, value: usize) {
        if !self.elements.contains(&value) {
            self.elements.push(value);
        }
    }

    fn remove(&mut self, value: &usize) {
        self.elements.retain(|&x| x != *value);
    }

    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&usize) -> bool,
    {
        self.elements.retain(f)
    }

    fn union(&mut self, rhs: &Self) {
        for &item in &rhs.elements {
            self.insert(item);
        }
    }

    fn symmetric_difference_update(&mut self, rhs: &Self) {
        let mut temp = Self::new();
        self.retain(|x| {
            if rhs.contains(x) {
                temp.insert(*x);
                false
            } else {
                true
            }
        });
        for item in &rhs.elements {
            if !temp.contains(item) {
                self.insert(*item);
            }
        }
    }

    type Iter<'a> = std::slice::Iter<'a, usize>;

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        self.elements.iter()
    }
}

impl BitXorAssign<&Self> for UVecSet {
    fn bitxor_assign(&mut self, rhs: &Self) {
        self.symmetric_difference_update(rhs);
    }
}

impl BitOrAssign<&Self> for UVecSet {
    fn bitor_assign(&mut self, rhs: &Self) {
        self.union(rhs);
    }
}

impl BitAndAssign<&Self> for UVecSet {
    fn bitand_assign(&mut self, rhs: &Self) {
        self.intersection(rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let set = UVecSet::new();
        assert!(set.elements.is_empty());
    }

    #[test]
    fn test_new_with_vec() {
        let set: UVecSet = vec![4, 5, 6, 4].into_iter().collect();
        assert_eq!(set.elements, vec![4, 5, 6]);
    }

    #[test]
    fn test_from() {
        let set = UVecSet::from([4, 5, 6, 4]);
        assert_eq!(set.elements, vec![4, 5, 6]);
    }

    #[test]
    fn test_insert() {
        let mut set = UVecSet::new();
        set.insert(1);
        assert_eq!(set.elements, vec![1]);
        set.insert(5);
        set.insert(1);
        assert_eq!(set.elements, vec![1, 5]);
    }

    #[test]
    fn test_remove() {
        let mut set = UVecSet::from([4, 5, 6, 4]);
        set.remove(&5);
        assert_eq!(set.elements, vec![4, 6]);
        set.remove(&7);
        assert_eq!(set.elements, vec![4, 6]);
    }

    #[test]
    fn test_union() {
        let mut set_a = UVecSet::from([1, 2]);
        let set_b = UVecSet::from([2, 3]);
        set_a.union(&set_b);
        assert_eq!(set_a.elements, vec![1, 2, 3]);
    }

    #[test]
    fn test_symmetric_difference_update() {
        let mut set_a = UVecSet::from([4, 5, 6, 4]);
        let set_b = UVecSet::from([1, 3, 4]);
        set_a.symmetric_difference(&set_b);
        assert_eq!(set_a.elements, vec![5, 6, 1, 3]);
    }

    #[test]
    fn test_bitor_assign() {
        let mut set_a = UVecSet::from([1, 2]);
        let set_b = UVecSet::from([2, 3]);
        set_a |= &set_b;
        assert_eq!(set_a.elements, vec![1, 2, 3])
    }

    #[test]
    fn test_bitxor_assign() {
        let mut set_a = UVecSet::from([1, 2, 3]);
        let set_b = UVecSet::from([2, 3, 4]);
        set_a ^= &set_b;
        assert_eq!(set_a.elements, vec![1, 4])
    }

    #[test]
    fn test_bitand_assign() {
        let mut set_a = UVecSet::from([1, 2, 3]);
        let set_b = UVecSet::from([2, 3, 4]);
        set_a &= &set_b;
        assert_eq!(set_a.elements, vec![2, 3])
    }
}
