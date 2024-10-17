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
use std::ops::{BitAndAssign, BitOrAssign, BitXorAssign};

use std::collections::HashSet;
use std::hash::Hash;

#[derive(Clone, Debug)]
pub struct UHashSet<T> {
    elements: HashSet<T>,
}

impl<T: Eq + Hash + Copy> Set<T> for UHashSet<T> {
    type Iter<'a> = std::collections::hash_set::Iter<'a, T> where T: 'a;

    fn new() -> UHashSet<T> {
        UHashSet {
            elements: HashSet::new(),
        }
    }

    fn contains(&self, value: &T) -> bool {
        self.elements.contains(value)
    }

    fn insert(&mut self, value: T) {
        self.elements.insert(value);
    }

    fn remove(&mut self, value: &T) {
        self.elements.remove(value);
    }

    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
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

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        self.elements.iter()
    }
}

impl<'a, T> IntoIterator for &'a UHashSet<T>
where
    T: Eq + Hash + Copy,
{
    type Item = &'a T;
    type IntoIter = std::collections::hash_set::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Eq + Hash + Copy> BitXorAssign<&Self> for UHashSet<T> {
    fn bitxor_assign(&mut self, rhs: &Self) {
        self.symmetric_difference_update(rhs);
    }
}

impl<T: Eq + Hash + Copy> BitOrAssign<&Self> for UHashSet<T> {
    fn bitor_assign(&mut self, rhs: &Self) {
        self.union(rhs);
    }
}

impl<T: Eq + Hash + Copy> BitAndAssign<&Self> for UHashSet<T> {
    fn bitand_assign(&mut self, rhs: &Self) {
        self.intersection(rhs);
    }
}
