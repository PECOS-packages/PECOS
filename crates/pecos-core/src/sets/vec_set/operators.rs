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

use crate::build_set_bit_ops;
use crate::Set;
use crate::VecSet;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Sub, SubAssign};

use crate::Element;

impl<'a, E: Element> BitAnd<&'a VecSet<E>> for &'a VecSet<E> {
    type Output = VecSet<E>;

    #[inline]
    fn bitand(self, rhs: &'a VecSet<E>) -> Self::Output {
        VecSet {
            elements: self.intersection(rhs).copied().collect(),
        }
    }
}

impl<'a, E: Element> BitOr<&'a VecSet<E>> for &VecSet<E> {
    type Output = VecSet<E>;

    #[inline]
    fn bitor(self, rhs: &'a VecSet<E>) -> VecSet<E> {
        VecSet {
            elements: self.union(rhs).copied().collect(),
        }
    }
}

impl<'a, E: Element> BitXor<&'a VecSet<E>> for &VecSet<E> {
    type Output = VecSet<E>;

    #[inline]
    fn bitxor(self, rhs: &'a VecSet<E>) -> VecSet<E> {
        VecSet {
            elements: self.symmetric_difference(rhs).copied().collect(),
        }
    }
}

impl<'a, E: Element> Sub<&'a VecSet<E>> for &VecSet<E> {
    type Output = VecSet<E>;

    #[inline]
    fn sub(self, rhs: &'a VecSet<E>) -> VecSet<E> {
        VecSet {
            elements: self.difference(rhs).copied().collect(),
        }
    }
}

// TODO: Shouldn't this be fore &VecSet<E>?
impl<E: Element> BitXorAssign<VecSet<E>> for VecSet<E> {
    #[inline]
    fn bitxor_assign(&mut self, rhs: VecSet<E>) {
        for item in rhs.elements {
            if self.contains(&item) {
                self.remove(&item);
            } else {
                self.insert(item);
            }
        }
    }
}

//impl<'a, T> BitXorAssign<&'a Self> for VecSet<T>
//where
//T: UnsignedInt + 'a,
//{
//fn bitxor_assign(&mut self, rhs: &Self) {
//self.bitxor_assign_set(rhs);
//}
//}

//impl<'a, T> BitXorAssign<&'a T> for VecSet<T>
//where
//T: UnsignedInt + 'a,
//{
//fn bitxor_assign(&mut self, rhs: &T) {
//self.bitxor_assign_item(rhs);
//}
//}

build_set_bit_ops!(VecSet<E>);
