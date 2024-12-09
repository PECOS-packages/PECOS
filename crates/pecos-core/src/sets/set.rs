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

use crate::Element;
use core::fmt::Debug;
use core::ops::{BitAndAssign, BitOrAssign, BitXorAssign, SubAssign};

pub trait Set<'a>:
    Debug
    + Clone
    + Default
    + BitAndAssign<&'a Self>
    + BitAndAssign<&'a Self::Element>
    + BitOrAssign<&'a Self>
    + BitOrAssign<&'a Self::Element>
    + BitXorAssign<&'a Self>
    + BitXorAssign<&'a Self::Element>
    + SubAssign<&'a Self>
    + SubAssign<&'a Self::Element>
where
    Self: 'a,
{
    type Element: Element + 'a;

    type Iter: Iterator<Item = &'a Self::Element>;
    type Difference: Iterator<Item = &'a Self::Element>;
    type Intersection: Iterator<Item = &'a Self::Element>;
    type SymmetricDifference: Iterator<Item = &'a Self::Element>;
    type Union: Iterator<Item = &'a Self::Element>;

    fn new() -> Self;

    fn capacity(&self) -> usize;
    fn clear(&mut self);
    fn contains(&self, value: &Self::Element) -> bool;
    /// Returns an iterator of elements representing the difference between this set and another,
    /// i.e., the elements that are in `self` but not in `other`.
    fn difference(&'a self, other: &'a Self) -> Self::Difference;
    fn difference_ref(&'a self, other: &'a Self) -> Self::Difference;
    fn insert(&mut self, value: Self::Element);
    // TODO: Make signature that matches HashSet
    fn intersection(&'a self, other: &'a Self) -> Self::Intersection;
    #[inline]
    fn intersection_update(&mut self, rhs: &Self) {
        self.retain(|&x| rhs.contains(&x));
    }
    #[inline]
    fn intersection_item_update(&mut self, value: &Self::Element) {
        self.retain(|&x| x == *value);
    }
    fn is_empty(&self) -> bool;
    fn iter(&'a self) -> Self::Iter;
    fn len(&self) -> usize;
    fn remove(&mut self, value: &Self::Element);
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&Self::Element) -> bool;
    fn symmetric_difference(&'a self, other: &'a Self) -> Self::SymmetricDifference;
    fn symmetric_difference_update(&mut self, rhs: &Self);

    #[inline]
    fn symmetric_difference_item_update(&mut self, value: &Self::Element) {
        if self.contains(value) {
            self.remove(value);
        } else {
            self.insert(*value);
        }
    }
    fn union(&'a self, other: &'a Self) -> Self::Union;
    fn union_update(&mut self, rhs: &Self);
    #[inline]
    fn union_item_update(&mut self, value: &Self::Element) {
        self.insert(*value);
    }
    fn with_capacity(capacity: usize) -> Self;

    #[inline]
    // Default implementations for BitAndAssign, BitOrAssign, BitXorAssign, and SubAssign
    fn bitand_assign_set(&mut self, rhs: &Self) {
        self.intersection_update(rhs);
    }

    #[inline]
    fn bitand_assign_item(&mut self, rhs: &Self::Element) {
        self.intersection_item_update(rhs);
    }

    #[inline]
    fn bitor_assign_set(&mut self, rhs: &Self) {
        self.union_update(rhs);
    }

    #[inline]
    fn bitor_assign_item(&mut self, rhs: &Self::Element) {
        self.union_item_update(rhs);
    }

    #[inline]
    fn bitxor_assign_set(&mut self, rhs: &Self) {
        self.symmetric_difference_update(rhs);
    }

    #[inline]
    fn bitxor_assign_item(&mut self, rhs: &Self::Element) {
        self.symmetric_difference_item_update(rhs);
    }

    #[inline]
    fn sub_assign_set(&mut self, rhs: &Self) {
        self.retain(|x| !rhs.contains(x));
    }

    #[inline]
    fn sub_assign_item(&mut self, rhs: &Self::Element) {
        self.remove(rhs);
    }
}

#[macro_export]
macro_rules! build_set_bit_ops {
    ($set_type:ty) => {
        impl<'a, E: Element> BitAndAssign<&'a $set_type> for $set_type {
            #[inline]
            fn bitand_assign(&mut self, rhs: &$set_type) {
                self.bitand_assign_set(rhs);
            }
        }

        impl<'a, E: Element> BitAndAssign<&'a E> for $set_type {
            #[inline]
            fn bitand_assign(&mut self, rhs: &E) {
                self.bitand_assign_item(rhs);
            }
        }

        impl<'a, E: Element> BitOrAssign<&'a $set_type> for $set_type {
            #[inline]
            fn bitor_assign(&mut self, rhs: &$set_type) {
                self.bitor_assign_set(rhs);
            }
        }

        impl<'a, E: Element> BitOrAssign<&'a E> for $set_type {
            #[inline]
            fn bitor_assign(&mut self, rhs: &E) {
                self.bitor_assign_item(rhs);
            }
        }

        impl<'a, E: Element> BitXorAssign<&'a $set_type> for $set_type {
            #[inline]
            fn bitxor_assign(&mut self, rhs: &$set_type) {
                self.bitxor_assign_set(rhs);
            }
        }

        impl<'a, E: Element> BitXorAssign<&'a E> for $set_type {
            #[inline]
            fn bitxor_assign(&mut self, rhs: &E) {
                self.bitxor_assign_item(rhs);
            }
        }

        impl<'a, E: Element> SubAssign<&'a $set_type> for $set_type {
            #[inline]
            fn sub_assign(&mut self, rhs: &$set_type) {
                self.sub_assign_set(rhs);
            }
        }

        impl<'a, E: Element> SubAssign<&'a E> for $set_type {
            #[inline]
            fn sub_assign(&mut self, rhs: &E) {
                self.sub_assign_item(rhs);
            }
        }
    };
}
