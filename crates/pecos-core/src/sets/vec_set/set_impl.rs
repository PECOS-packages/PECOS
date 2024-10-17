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

use crate::sets::vec_set::iterators::{Difference, Intersection, SymmetricDifference, Union};
use crate::VecSet;
use core::slice::Iter;

use crate::{Element, Set};

impl<'a, E: Element + 'a> Set<'a> for VecSet<E> {
    type Element = E;
    type Iter = Iter<'a, E>;
    type Difference = Difference<'a, E>;
    type Intersection = Intersection<'a, E>;
    type SymmetricDifference = SymmetricDifference<'a, E>;
    type Union = Union<'a, E>;

    #[inline]
    #[must_use]
    fn new() -> Self {
        Self::new()
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.elements.capacity()
    }

    #[inline]
    fn clear(&mut self) {
        self.elements.clear();
    }

    #[inline]
    fn contains(&self, value: &E) -> bool {
        self.elements.contains(value)
    }

    #[inline]
    fn difference(&'a self, other: &'a Self) -> Self::Difference {
        Difference {
            iter: self.elements.iter(),
            other,
        }
    }

    #[inline]
    fn difference_ref(&'a self, other: &'a Self) -> Self::Difference {
        self.difference(other)
    }

    #[inline]
    fn insert(&mut self, value: E) {
        if !self.elements.contains(&value) {
            self.elements.push(value);
        }
    }

    #[inline]
    fn intersection(&'a self, other: &'a Self) -> Self::Intersection {
        if self.len() <= other.len() {
            Intersection {
                iter: self.iter(),
                other,
            }
        } else {
            Intersection {
                iter: other.iter(),
                other: self,
            }
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    #[inline]
    fn iter(&'a self) -> Self::Iter {
        self.elements.iter()
    }

    #[inline]
    fn len(&self) -> usize {
        self.elements.len()
    }

    #[inline]
    fn remove(&mut self, value: &E) {
        self.elements.retain(|&x| x != *value);
    }

    #[inline]
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&E) -> bool,
    {
        self.elements.retain(f);
    }

    #[inline]
    fn symmetric_difference(&'a self, other: &'a Self) -> Self::SymmetricDifference {
        SymmetricDifference {
            iter: self.difference(other).chain(other.difference(self)),
        }
    }

    #[inline]
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

    #[inline]
    fn union(&'a self, other: &'a Self) -> Self::Union {
        if self.len() >= other.len() {
            Union {
                iter: self.iter().chain(other.difference(self)),
            }
        } else {
            Union {
                iter: other.iter().chain(self.difference(other)),
            }
        }
    }

    #[inline]
    fn union_update(&mut self, rhs: &Self) {
        for &item in &rhs.elements {
            self.insert(item);
        }
    }

    #[inline]
    #[must_use]
    fn with_capacity(capacity: usize) -> Self {
        Self {
            elements: Vec::with_capacity(capacity),
        }
    }
}
