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

use core::fmt::Debug;
use core::hash::Hash;

pub trait Element: Debug + Clone + Ord + Copy + Hash {}

// blanket implementation should allow for integers, char, etc.
impl<T: Debug + Clone + Ord + Copy + Hash> Element for T {}

#[allow(clippy::module_name_repetitions)]
pub trait IndexableElement: Element {
    fn to_usize(&self) -> usize;
    fn from_usize(value: usize) -> Self;
}

macro_rules! impl_indexable_element_safe {
    ($t:ty) => {
        impl IndexableElement for $t {
            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            #[inline(always)]
            fn to_usize(&self) -> usize {
                *self as usize
            }

            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            #[inline(always)]
            fn from_usize(value: usize) -> Self {
                value as $t
            }
        }
    };
}

// Safe implementations for types that are always smaller than or equal to usize
impl_indexable_element_safe!(u8);
impl_indexable_element_safe!(u16);
impl_indexable_element_safe!(u32);
impl_indexable_element_safe!(usize);

// Conditional implementation for u64
#[cfg(target_pointer_width = "64")]
impl_indexable_element_safe!(u64);

#[cfg(target_pointer_width = "32")]
impl IndexableElement for u64 {
    #[inline(always)]
    fn to_usize(&self) -> usize {
        usize::try_from(*self).expect("u64 value too large for 32-bit usize")
    }

    #[allow(clippy::as_conversions)]
    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        value as u64
    }
}

// u128 is always problematic for current architectures, so we'll implement it to always panic
impl IndexableElement for u128 {
    #[inline(always)]
    fn to_usize(&self) -> usize {
        panic!("u128 cannot be safely converted to usize without potential data loss")
    }

    #[inline(always)]
    fn from_usize(_value: usize) -> Self {
        panic!("usize cannot be safely converted to u128 on all platforms")
    }
}
