// Copyright 2025 The PECOS Developers
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

//! A fixed-width unsigned integer with explicit bit width tracking.
//!
//! [`BitUInt`] is the primitive unsigned N-bit integer type. All bit manipulation
//! logic lives here. `BitInt` wraps `BitUInt(N+1)` to provide signed semantics.
//!
//! # Examples
//!
//! ```
//! use pecos_core::BitUInt;
//!
//! let a = BitUInt::new(8, 0b1010_1010);
//! let b = BitUInt::new(8, 0b0101_0101);
//! let c = &a ^ &b;
//! assert_eq!(c.to_u64(), Some(0xFF));
//! ```

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Not, Rem, Shl, Shr, Sub};

/// Internal storage for `BitUInt` values.
#[derive(Clone, Debug, PartialEq, Eq)]
enum BitUIntValue {
    /// Fast path: single 64-bit word for widths <= 64
    Small(u64),
    /// Arbitrary precision: packed u64 words, LSB first
    Large(Box<[u64]>),
}

/// A fixed-width unsigned integer with explicit bit width tracking.
///
/// Values are always masked to the specified bit width after operations.
/// Shift right is always logical (fills with 0). Division and remainder
/// are always unsigned.
#[derive(Clone, Debug)]
pub struct BitUInt {
    /// Bit width of this integer (1 to 65535)
    size: u16,
    /// The actual value storage
    value: BitUIntValue,
}

impl BitUInt {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a new `BitUInt` with the given size and value.
    ///
    /// The value is masked to fit within the specified bit width.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    #[must_use]
    pub fn new(size: u16, value: u64) -> Self {
        assert!(size > 0, "BitUInt size must be at least 1");
        let mut result = Self {
            size,
            value: if size <= 64 {
                BitUIntValue::Small(value)
            } else {
                let num_words = Self::words_needed(size);
                let mut words = vec![0u64; num_words].into_boxed_slice();
                words[0] = value;
                BitUIntValue::Large(words)
            },
        };
        result.mask_to_width();
        result
    }

    /// Create a `BitUInt` from raw word data (LSB first).
    ///
    /// Words are in little-endian order (least significant word first).
    /// Missing words are zero-padded and excess words are discarded.
    /// The value is masked to fit within the specified bit width.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    #[must_use]
    pub fn from_raw_words(size: u16, words: Box<[u64]>) -> Self {
        assert!(size > 0, "BitUInt size must be at least 1");
        let mut result = Self {
            size,
            value: if size <= 64 {
                BitUIntValue::Small(words.first().copied().unwrap_or(0))
            } else {
                let mut words = words.into_vec();
                words.resize(Self::words_needed(size), 0);
                BitUIntValue::Large(words.into_boxed_slice())
            },
        };
        result.mask_to_width();
        result
    }

    /// Create a zero value with the given size.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    #[must_use]
    pub fn zero(size: u16) -> Self {
        assert!(size > 0, "BitUInt size must be at least 1");
        Self {
            size,
            value: if size <= 64 {
                BitUIntValue::Small(0)
            } else {
                let num_words = Self::words_needed(size);
                BitUIntValue::Large(vec![0u64; num_words].into_boxed_slice())
            },
        }
    }

    /// Create an all-ones value with the given size.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    #[must_use]
    pub fn ones(size: u16) -> Self {
        assert!(size > 0, "BitUInt size must be at least 1");
        let mut result = Self {
            size,
            value: if size <= 64 {
                BitUIntValue::Small(u64::MAX)
            } else {
                let num_words = Self::words_needed(size);
                BitUIntValue::Large(vec![u64::MAX; num_words].into_boxed_slice())
            },
        };
        result.mask_to_width();
        result
    }

    /// Create a `BitUInt` from a binary string.
    ///
    /// The size is determined by the string length.
    ///
    /// # Panics
    ///
    /// Panics if the string is empty, exceeds 65535 characters, or contains
    /// non-binary characters.
    #[must_use]
    pub fn from_binary_str(s: &str) -> Self {
        assert!(!s.is_empty(), "Binary string must not be empty");
        let size = u16::try_from(s.len()).expect("Binary string too long (max 65535 chars)");
        assert!(
            s.bytes().all(|bit| bit == b'0' || bit == b'1'),
            "Invalid binary string"
        );

        if size <= 64 {
            let value = u64::from_str_radix(s, 2).expect("Invalid binary string");
            Self::new(size, value)
        } else {
            let mut words = Vec::with_capacity(Self::words_needed(size));
            // Parse from the least significant end, including a partial top word.
            let mut chunk_end = s.len();
            while chunk_end > 0 {
                let chunk_start = chunk_end.saturating_sub(64);
                let word = u64::from_str_radix(&s[chunk_start..chunk_end], 2)
                    .expect("Invalid binary string");
                words.push(word);
                chunk_end = chunk_start;
            }

            Self::from_raw_words(size, words.into_boxed_slice())
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Returns the bit width of this integer.
    #[must_use]
    pub fn size(&self) -> u16 {
        self.size
    }

    /// Always returns false (unsigned).
    #[must_use]
    pub fn is_signed(&self) -> bool {
        false
    }

    /// Returns the value as a `u64` if it fits, otherwise `None`.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        match &self.value {
            BitUIntValue::Small(v) => Some(*v),
            BitUIntValue::Large(words) => {
                if words.iter().skip(1).all(|&w| w == 0) {
                    Some(words[0])
                } else {
                    None
                }
            }
        }
    }

    /// Returns the value as an `i64`. This is a simple cast from `u64`.
    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        self.to_u64().map(|v| {
            #[allow(clippy::cast_possible_wrap)]
            let result = v as i64;
            result
        })
    }

    /// Get the value of a specific bit (0-indexed from LSB).
    ///
    /// # Panics
    ///
    /// Panics if `index >= size`.
    #[must_use]
    pub fn get_bit(&self, index: u16) -> bool {
        assert!(index < self.size, "Bit index out of bounds");
        match &self.value {
            BitUIntValue::Small(v) => (*v >> index) & 1 == 1,
            BitUIntValue::Large(words) => {
                let word_idx = (index / 64) as usize;
                let bit_idx = index % 64;
                (words[word_idx] >> bit_idx) & 1 == 1
            }
        }
    }

    /// Set the value of a specific bit (0-indexed from LSB).
    ///
    /// # Panics
    ///
    /// Panics if `index >= size`.
    pub fn set_bit(&mut self, index: u16, value: bool) {
        assert!(index < self.size, "Bit index out of bounds");
        match &mut self.value {
            BitUIntValue::Small(v) => {
                if value {
                    *v |= 1 << index;
                } else {
                    *v &= !(1 << index);
                }
            }
            BitUIntValue::Large(words) => {
                let word_idx = (index / 64) as usize;
                let bit_idx = index % 64;
                if value {
                    words[word_idx] |= 1 << bit_idx;
                } else {
                    words[word_idx] &= !(1 << bit_idx);
                }
            }
        }
    }

    /// Returns the number of 1 bits (population count).
    #[must_use]
    pub fn count_ones(&self) -> u32 {
        match &self.value {
            BitUIntValue::Small(v) => v.count_ones(),
            BitUIntValue::Large(words) => words.iter().map(|w| w.count_ones()).sum(),
        }
    }

    /// Returns the number of 0 bits.
    #[must_use]
    pub fn count_zeros(&self) -> u32 {
        u32::from(self.size) - self.count_ones()
    }

    /// Returns the value as a vector of u64 words in little-endian order (LSB first).
    #[must_use]
    pub fn to_words(&self) -> Vec<u64> {
        match &self.value {
            BitUIntValue::Small(v) => vec![*v],
            BitUIntValue::Large(words) => words.to_vec(),
        }
    }

    /// Returns true if the value is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        match &self.value {
            BitUIntValue::Small(v) => *v == 0,
            BitUIntValue::Large(words) => words.iter().all(|&w| w == 0),
        }
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Calculate the number of 64-bit words needed for a given bit width.
    #[must_use]
    fn words_needed(size: u16) -> usize {
        (size as usize).div_ceil(64)
    }

    /// Mask the value to fit within the bit width.
    fn mask_to_width(&mut self) {
        match &mut self.value {
            BitUIntValue::Small(v) => {
                if self.size < 64 {
                    *v &= (1u64 << self.size) - 1;
                }
            }
            BitUIntValue::Large(words) => {
                let last_word_bits = self.size % 64;
                if last_word_bits > 0 {
                    let last_idx = words.len() - 1;
                    words[last_idx] &= (1u64 << last_word_bits) - 1;
                }
            }
        }
    }

    /// Get the raw underlying u64 value (first word).
    #[must_use]
    pub(crate) fn raw_u64(&self) -> u64 {
        match &self.value {
            BitUIntValue::Small(v) => *v,
            BitUIntValue::Large(words) => words[0],
        }
    }

    /// Get word at index, or 0 if beyond bounds.
    #[must_use]
    fn word_at(&self, index: usize) -> u64 {
        match &self.value {
            BitUIntValue::Small(v) => {
                if index == 0 {
                    *v
                } else {
                    0
                }
            }
            BitUIntValue::Large(words) => words.get(index).copied().unwrap_or(0),
        }
    }

    /// Create a new `BitUInt` with the same size as self, with the given small value.
    fn new_with_same_size(&self, value: u64) -> Self {
        Self::new(self.size, value)
    }

    /// Create a new `BitUInt` with the same size as self, with large value.
    fn new_with_same_size_large(&self, words: Box<[u64]>) -> Self {
        Self::from_raw_words(self.size, words)
    }

    /// Unsigned shift-and-add kernel, the inner loop of
    /// [`crate::bitvec::arithmetic::multiply`] without its sign handling.
    fn multiply(&self, rhs: &Self) -> Self {
        let mut result = Self::zero(self.size);
        for i in 0..self.size.min(rhs.size) {
            if rhs.get_bit(i) {
                result = &result + &(self << i);
            }
        }
        result
    }

    /// Unsigned long-division kernel, the inner loop of
    /// [`crate::bitvec::arithmetic::divide`] without its sign handling.
    /// The caller must reject a zero divisor.
    fn div_rem(&self, rhs: &Self) -> (Self, Self) {
        debug_assert!(!rhs.is_zero(), "div_rem requires a nonzero divisor");
        if self.size <= 64 && rhs.size <= 64 {
            return (
                Self::new(self.size, self.raw_u64() / rhs.raw_u64()),
                Self::new(self.size, self.raw_u64() % rhs.raw_u64()),
            );
        }

        let mut quotient = Self::zero(self.size);
        let mut remainder = self.clone();
        // Compare the full divisor before resizing: a wider divisor may not fit.
        if self < rhs {
            return (quotient, remainder);
        }
        let divisor = Self::from_raw_words(self.size, rhs.to_words().into_boxed_slice());
        let divisor_bits = (0..rhs.size)
            .rev()
            .find(|&i| rhs.get_bit(i))
            .expect("Nonzero divisor")
            + 1;
        // The early return above leaves rhs <= self < 2^size, so the divisor's bit
        // length cannot exceed the dividend's width and the subtraction is safe.
        debug_assert!(
            divisor_bits <= self.size,
            "divisor bit length exceeds dividend width"
        );
        // Every shifted divisor fits at this width, so no high bits are lost.
        for shift in (0..=self.size - divisor_bits).rev() {
            let shifted = &divisor << shift;
            if remainder >= shifted {
                remainder = &remainder - &shifted;
                quotient.set_bit(shift, true);
            }
        }
        (quotient, remainder)
    }
}

// ============================================================================
// Display
// ============================================================================

impl fmt::Display for BitUInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.value {
            BitUIntValue::Small(v) => {
                write!(f, "{:0>width$b}", v, width = self.size as usize)
            }
            BitUIntValue::Large(_) => {
                let mut s = String::with_capacity(self.size as usize);
                for i in (0..self.size).rev() {
                    s.push(if self.get_bit(i) { '1' } else { '0' });
                }
                write!(f, "{s}")
            }
        }
    }
}

// ============================================================================
// Equality and Ordering
// ============================================================================

impl PartialEq for BitUInt {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for BitUInt {}

impl PartialOrd for BitUInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BitUInt {
    fn cmp(&self, other: &Self) -> Ordering {
        for i in (0..Self::words_needed(self.size.max(other.size))).rev() {
            let ordering = self.word_at(i).cmp(&other.word_at(i));
            if ordering != Ordering::Equal {
                return ordering;
            }
        }
        Ordering::Equal
    }
}

// ============================================================================
// Bitwise Operations
// ============================================================================

impl BitXor for &BitUInt {
    type Output = BitUInt;

    fn bitxor(self, rhs: Self) -> BitUInt {
        match &self.value {
            BitUIntValue::Small(_) => self.new_with_same_size(self.raw_u64() ^ rhs.raw_u64()),
            BitUIntValue::Large(words) => {
                let result: Box<[u64]> = words
                    .iter()
                    .enumerate()
                    .map(|(i, &w)| w ^ rhs.word_at(i))
                    .collect();
                self.new_with_same_size_large(result)
            }
        }
    }
}

impl BitAnd for &BitUInt {
    type Output = BitUInt;

    fn bitand(self, rhs: Self) -> BitUInt {
        match &self.value {
            BitUIntValue::Small(_) => self.new_with_same_size(self.raw_u64() & rhs.raw_u64()),
            BitUIntValue::Large(words) => {
                let result: Box<[u64]> = words
                    .iter()
                    .enumerate()
                    .map(|(i, &w)| w & rhs.word_at(i))
                    .collect();
                self.new_with_same_size_large(result)
            }
        }
    }
}

impl BitOr for &BitUInt {
    type Output = BitUInt;

    fn bitor(self, rhs: Self) -> BitUInt {
        match &self.value {
            BitUIntValue::Small(_) => self.new_with_same_size(self.raw_u64() | rhs.raw_u64()),
            BitUIntValue::Large(words) => {
                let result: Box<[u64]> = words
                    .iter()
                    .enumerate()
                    .map(|(i, &w)| w | rhs.word_at(i))
                    .collect();
                self.new_with_same_size_large(result)
            }
        }
    }
}

impl Not for &BitUInt {
    type Output = BitUInt;

    fn not(self) -> BitUInt {
        match &self.value {
            BitUIntValue::Small(v) => self.new_with_same_size(!v),
            BitUIntValue::Large(words) => {
                let new_words: Box<[u64]> = words.iter().map(|w| !w).collect();
                self.new_with_same_size_large(new_words)
            }
        }
    }
}

// ============================================================================
// Shift Operations (always logical)
// ============================================================================

impl Shl<u16> for &BitUInt {
    type Output = BitUInt;

    fn shl(self, rhs: u16) -> BitUInt {
        if rhs >= self.size {
            return BitUInt::zero(self.size);
        }

        match &self.value {
            BitUIntValue::Small(v) => self.new_with_same_size(v << rhs),
            BitUIntValue::Large(words) => {
                let word_shift = (rhs / 64) as usize;
                let bit_shift = rhs % 64;

                let mut new_words = vec![0u64; words.len()];

                for i in word_shift..words.len() {
                    new_words[i] = words[i - word_shift] << bit_shift;
                    if bit_shift > 0 && i > word_shift {
                        new_words[i] |= words[i - word_shift - 1] >> (64 - bit_shift);
                    }
                }

                self.new_with_same_size_large(new_words.into_boxed_slice())
            }
        }
    }
}

impl Shr<u16> for &BitUInt {
    type Output = BitUInt;

    /// Logical shift right (always fills with 0).
    fn shr(self, rhs: u16) -> BitUInt {
        if rhs >= self.size {
            return BitUInt::zero(self.size);
        }

        match &self.value {
            BitUIntValue::Small(v) => self.new_with_same_size(v >> rhs),
            BitUIntValue::Large(words) => {
                let word_shift = (rhs / 64) as usize;
                let bit_shift = rhs % 64;

                let mut new_words = vec![0u64; words.len()];

                for i in 0..(words.len() - word_shift) {
                    new_words[i] = words[i + word_shift] >> bit_shift;
                    if bit_shift > 0 && i + word_shift + 1 < words.len() {
                        new_words[i] |= words[i + word_shift + 1] << (64 - bit_shift);
                    }
                }

                self.new_with_same_size_large(new_words.into_boxed_slice())
            }
        }
    }
}

// ============================================================================
// Arithmetic Operations (always unsigned)
// ============================================================================

impl Add for &BitUInt {
    type Output = BitUInt;

    fn add(self, rhs: Self) -> BitUInt {
        match &self.value {
            BitUIntValue::Small(_) => {
                self.new_with_same_size(self.raw_u64().wrapping_add(rhs.raw_u64()))
            }
            BitUIntValue::Large(words) => {
                let mut result = vec![0u64; words.len()];
                let mut carry = 0u64;

                for i in 0..words.len() {
                    let (sum1, c1) = words[i].overflowing_add(rhs.word_at(i));
                    let (sum2, c2) = sum1.overflowing_add(carry);
                    result[i] = sum2;
                    carry = u64::from(c1) + u64::from(c2);
                }

                self.new_with_same_size_large(result.into_boxed_slice())
            }
        }
    }
}

impl Sub for &BitUInt {
    type Output = BitUInt;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> BitUInt {
        match &self.value {
            BitUIntValue::Small(_) => {
                self.new_with_same_size(self.raw_u64().wrapping_sub(rhs.raw_u64()))
            }
            BitUIntValue::Large(words) => {
                let mut result = vec![0u64; words.len()];
                let mut borrow = 0u64;

                for i in 0..words.len() {
                    let (diff1, b1) = words[i].overflowing_sub(rhs.word_at(i));
                    let (diff2, b2) = diff1.overflowing_sub(borrow);
                    result[i] = diff2;
                    borrow = u64::from(b1) + u64::from(b2);
                }

                self.new_with_same_size_large(result.into_boxed_slice())
            }
        }
    }
}

impl Mul for &BitUInt {
    type Output = BitUInt;

    fn mul(self, rhs: Self) -> BitUInt {
        match &self.value {
            BitUIntValue::Small(_) => {
                self.new_with_same_size(self.raw_u64().wrapping_mul(rhs.raw_u64()))
            }
            BitUIntValue::Large(_) => self.multiply(rhs),
        }
    }
}

impl Div for &BitUInt {
    type Output = BitUInt;

    /// Always unsigned division.
    fn div(self, rhs: Self) -> BitUInt {
        assert!(!rhs.is_zero(), "Division by zero");
        self.div_rem(rhs).0
    }
}

impl Rem for &BitUInt {
    type Output = BitUInt;

    /// Always unsigned remainder.
    fn rem(self, rhs: Self) -> BitUInt {
        assert!(!rhs.is_zero(), "Remainder by zero");
        self.div_rem(rhs).1
    }
}

// ============================================================================
// Owned value operations (forward to reference implementations)
// ============================================================================

macro_rules! impl_binop_owned {
    ($trait:ident, $method:ident) => {
        impl $trait for BitUInt {
            type Output = BitUInt;
            fn $method(self, rhs: Self) -> BitUInt {
                (&self).$method(&rhs)
            }
        }
        impl $trait<&BitUInt> for BitUInt {
            type Output = BitUInt;
            fn $method(self, rhs: &BitUInt) -> BitUInt {
                (&self).$method(rhs)
            }
        }
        impl $trait<BitUInt> for &BitUInt {
            type Output = BitUInt;
            fn $method(self, rhs: BitUInt) -> BitUInt {
                self.$method(&rhs)
            }
        }
    };
}

impl_binop_owned!(BitXor, bitxor);
impl_binop_owned!(BitAnd, bitand);
impl_binop_owned!(BitOr, bitor);
impl_binop_owned!(Add, add);
impl_binop_owned!(Sub, sub);
impl_binop_owned!(Mul, mul);
impl_binop_owned!(Div, div);
impl_binop_owned!(Rem, rem);

impl Not for BitUInt {
    type Output = BitUInt;
    fn not(self) -> BitUInt {
        (&self).not()
    }
}

impl Shl<u16> for BitUInt {
    type Output = BitUInt;
    fn shl(self, rhs: u16) -> BitUInt {
        (&self).shl(rhs)
    }
}

impl Shr<u16> for BitUInt {
    type Output = BitUInt;
    fn shr(self, rhs: u16) -> BitUInt {
        (&self).shr(rhs)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let a = BitUInt::new(8, 0xFF);
        assert_eq!(a.size(), 8);
        assert!(!a.is_signed());
        assert_eq!(a.to_u64(), Some(0xFF));
    }

    #[test]
    fn test_masking() {
        let a = BitUInt::new(4, 0xFF);
        assert_eq!(a.to_u64(), Some(0x0F));
    }

    #[test]
    fn test_1bit_returns_positive() {
        let a = BitUInt::new(1, 1);
        assert_eq!(a.to_u64(), Some(1));
        assert_eq!(a.to_i64(), Some(1));
    }

    #[test]
    fn test_zero() {
        let a = BitUInt::zero(8);
        assert_eq!(a.to_u64(), Some(0));
        assert!(a.is_zero());
    }

    #[test]
    fn test_ones() {
        let a = BitUInt::ones(8);
        assert_eq!(a.to_u64(), Some(0xFF));
    }

    #[test]
    fn test_64bit() {
        let a = BitUInt::new(64, u64::MAX);
        assert_eq!(a.to_u64(), Some(u64::MAX));
    }

    #[test]
    #[should_panic(expected = "BitUInt size must be at least 1")]
    fn test_reject_size_0() {
        let _ = BitUInt::new(0, 0);
    }

    #[test]
    fn test_bit_access() {
        let mut a = BitUInt::new(8, 0b1010_0101);
        assert!(a.get_bit(0));
        assert!(!a.get_bit(1));
        assert!(a.get_bit(2));

        a.set_bit(1, true);
        assert!(a.get_bit(1));
        assert_eq!(a.to_u64(), Some(0b1010_0111));
    }

    #[test]
    fn test_bitwise_xor() {
        let a = BitUInt::new(8, 0b1010_1010);
        let b = BitUInt::new(8, 0b0101_0101);
        let c = &a ^ &b;
        assert_eq!(c.to_u64(), Some(0xFF));
    }

    #[test]
    fn test_bitwise_and() {
        let a = BitUInt::new(8, 0b1010_1010);
        let b = BitUInt::new(8, 0b1111_0000);
        let c = &a & &b;
        assert_eq!(c.to_u64(), Some(0b1010_0000));
    }

    #[test]
    fn test_bitwise_or() {
        let a = BitUInt::new(8, 0b1010_0000);
        let b = BitUInt::new(8, 0b0000_0101);
        let c = &a | &b;
        assert_eq!(c.to_u64(), Some(0b1010_0101));
    }

    #[test]
    fn test_bitwise_not() {
        let a = BitUInt::new(8, 0b1010_1010);
        let b = !&a;
        assert_eq!(b.to_u64(), Some(0b0101_0101));
    }

    #[test]
    fn test_shift_left() {
        let a = BitUInt::new(8, 0b0000_1111);
        let b = &a << 4;
        assert_eq!(b.to_u64(), Some(0b1111_0000));
    }

    #[test]
    fn test_shift_right_logical() {
        let a = BitUInt::new(8, 0b1111_0000);
        let b = &a >> 4;
        assert_eq!(b.to_u64(), Some(0b0000_1111));
    }

    #[test]
    fn test_add() {
        let a = BitUInt::new(8, 100);
        let b = BitUInt::new(8, 50);
        let c = &a + &b;
        assert_eq!(c.to_u64(), Some(150));
    }

    #[test]
    fn test_add_overflow() {
        let a = BitUInt::new(8, 200);
        let b = BitUInt::new(8, 100);
        let c = &a + &b;
        assert_eq!(c.to_u64(), Some(44)); // (200+100) % 256 = 44
    }

    #[test]
    fn test_sub() {
        let a = BitUInt::new(8, 100);
        let b = BitUInt::new(8, 50);
        let c = &a - &b;
        assert_eq!(c.to_u64(), Some(50));
    }

    #[test]
    fn test_sub_underflow() {
        let a = BitUInt::new(8, 5);
        let b = BitUInt::new(8, 10);
        let c = &a - &b;
        assert_eq!(c.to_u64(), Some(251)); // wraps: (5 - 10) % 256
    }

    #[test]
    fn test_mul() {
        let a = BitUInt::new(8, 10);
        let b = BitUInt::new(8, 5);
        let c = &a * &b;
        assert_eq!(c.to_u64(), Some(50));
    }

    #[test]
    fn test_div() {
        let a = BitUInt::new(8, 100);
        let b = BitUInt::new(8, 10);
        let c = &a / &b;
        assert_eq!(c.to_u64(), Some(10));
    }

    #[test]
    fn test_rem() {
        let a = BitUInt::new(8, 100);
        let b = BitUInt::new(8, 30);
        let c = &a % &b;
        assert_eq!(c.to_u64(), Some(10));
    }

    #[test]
    fn test_comparison() {
        let a = BitUInt::new(8, 100);
        let b = BitUInt::new(8, 50);
        let c = BitUInt::new(8, 100);

        assert!(a > b);
        assert!(b < a);
        assert_eq!(a, c);
    }

    #[test]
    fn test_count_ones() {
        let a = BitUInt::new(8, 0b1010_1010);
        assert_eq!(a.count_ones(), 4);
    }

    #[test]
    fn test_count_zeros() {
        let a = BitUInt::new(8, 0b1010_1010);
        assert_eq!(a.count_zeros(), 4);
    }

    #[test]
    fn test_display() {
        let a = BitUInt::new(8, 0b1010_0101);
        assert_eq!(format!("{a}"), "10100101");

        let b = BitUInt::new(4, 0b0101);
        assert_eq!(format!("{b}"), "0101");
    }

    #[test]
    fn test_from_binary_str() {
        let a = BitUInt::from_binary_str("1010");
        assert_eq!(a.size(), 4);
        assert_eq!(a.to_u64(), Some(0b1010));
    }

    #[test]
    fn test_large_bituint() {
        let a = BitUInt::new(128, 0xFFFF_FFFF_FFFF_FFFF);
        assert_eq!(a.size(), 128);
        assert_eq!(a.to_u64(), Some(0xFFFF_FFFF_FFFF_FFFF));
        assert!(a.get_bit(0));
        assert!(a.get_bit(63));
        assert!(!a.get_bit(64));
    }

    #[test]
    fn test_mixed_size_xor() {
        let a = BitUInt::new(8, 0b1010_1010);
        let b = BitUInt::new(4, 0b0101);
        let c = &a ^ &b;
        assert_eq!(c.size(), 8);
        assert_eq!(c.to_u64(), Some(0b1010_1111));
    }

    #[test]
    fn test_mixed_size_and() {
        let a = BitUInt::new(8, 0b1111_1111);
        let b = BitUInt::new(4, 0b1010);
        let c = &a & &b;
        assert_eq!(c.size(), 8);
        assert_eq!(c.to_u64(), Some(0b0000_1010));
    }

    #[test]
    fn test_mixed_size_add() {
        let a = BitUInt::new(8, 200);
        let b = BitUInt::new(4, 10);
        let c = &a + &b;
        assert_eq!(c.size(), 8);
        assert_eq!(c.to_u64(), Some(210));
    }

    #[test]
    fn test_mixed_size_comparison() {
        let a = BitUInt::new(8, 10);
        let b = BitUInt::new(4, 10);
        assert_eq!(a, b);

        let c = BitUInt::new(8, 20);
        assert!(c > b);
    }

    // Packet reference values are encoded by their supplied set-bit lists or words.
    fn with_bits(size: u16, bits: &[u16]) -> BitUInt {
        let mut value = BitUInt::zero(size);
        for &bit in bits {
            value.set_bit(bit, true);
        }
        value
    }

    fn assert_words(value: &BitUInt, size: u16, words: &[u64]) {
        assert_eq!(value.size(), size);
        assert_eq!(value.to_words(), words);
    }

    #[test]
    fn w100_add() {
        let a = with_bits(100, &[0, 50, 99]);
        let b = with_bits(100, &[0, 50, 98]);
        assert_words(
            &(&a + &b),
            100,
            &with_bits(100, &[1, 51, 98, 99]).to_words(),
        );
    }

    #[test]
    fn w100_sub() {
        let a = with_bits(100, &[0, 50, 99]);
        let b = with_bits(100, &[0, 50, 98]);
        assert_words(&(&a - &b), 100, &with_bits(100, &[98]).to_words());
    }

    #[test]
    fn w100_mul() {
        let a = with_bits(100, &[0, 50, 99]);
        let b = with_bits(100, &[0, 50, 98]);
        assert_words(
            &(&a * &b),
            100,
            &with_bits(100, &[0, 51, 98, 99]).to_words(),
        );
    }

    #[test]
    fn w100_div() {
        let a = with_bits(100, &[0, 50, 99]);
        let b = with_bits(100, &[0, 50, 98]);
        assert_words(&(&a / &b), 100, &with_bits(100, &[0]).to_words());
    }

    #[test]
    fn w100_rem() {
        let a = with_bits(100, &[0, 50, 99]);
        let b = with_bits(100, &[0, 50, 98]);
        assert_words(&(&a % &b), 100, &with_bits(100, &[98]).to_words());
    }

    #[test]
    fn w65_divisor_with_zero_low_word_div() {
        let a = with_bits(65, &[0, 2, 64]);
        let b = with_bits(65, &[64]);
        assert_words(&(&a / &b), 65, &with_bits(65, &[0]).to_words());
    }

    #[test]
    fn w65_divisor_with_zero_low_word_rem() {
        let a = with_bits(65, &[0, 2, 64]);
        let b = with_bits(65, &[64]);
        assert_words(&(&a % &b), 65, &with_bits(65, &[0, 2]).to_words());
    }

    #[test]
    fn w128_max_div_three() {
        let a = with_bits(128, &(0..127).collect::<Vec<_>>());
        let b = with_bits(128, &[0, 1]);
        assert_words(
            &(&a / &b),
            128,
            &with_bits(128, &(1..126).step_by(2).collect::<Vec<_>>()).to_words(),
        );
    }

    #[test]
    fn w128_max_rem_three() {
        let a = with_bits(128, &(0..127).collect::<Vec<_>>());
        let b = with_bits(128, &[0, 1]);
        assert_words(&(&a % &b), 128, &with_bits(128, &[0]).to_words());
    }

    #[test]
    fn w128_max_times_three() {
        let a = with_bits(128, &(0..127).collect::<Vec<_>>());
        let b = with_bits(128, &[0, 1]);
        assert_words(
            &(&a * &b),
            128,
            &with_bits(128, &std::iter::once(0).chain(2..127).collect::<Vec<_>>()).to_words(),
        );
    }

    #[test]
    fn w128_carry_above_low_word() {
        let a = with_bits(128, &[63]);
        let b = with_bits(128, &[1]);
        assert_words(&(&a * &b), 128, &[0, 1]);
    }

    #[test]
    fn w200_square() {
        let a = with_bits(200, &[0, 100]);
        let b = with_bits(200, &[0, 100]);
        assert_words(&(&a * &b), 200, &with_bits(200, &[0, 101]).to_words());
    }

    #[test]
    fn w128_product_wrapping_past_the_top_bit() {
        // A zero expectation alone would also hold for a multiply that returned
        // zero, so pair it with a wrap that keeps bits: (2^64+1)^2 is
        // 2^128 + 2^65 + 1, which is 2^65 + 1 at this width.
        let a = with_bits(128, &[64]);
        let b = with_bits(128, &[64]);
        assert_words(&(&a * &b), 128, &with_bits(128, &[]).to_words());

        let c = with_bits(128, &[0, 64]);
        assert_words(&(&c * &c), 128, &with_bits(128, &[0, 65]).to_words());
    }

    #[test]
    fn w200_quotient_keeps_a_bit_above_127() {
        let a = with_bits(200, &[170]);
        let b = with_bits(200, &[5]);
        assert_words(&(&a / &b), 200, &with_bits(200, &[165]).to_words());
        assert_words(&(&a % &b), 200, &with_bits(200, &[]).to_words());
    }

    #[test]
    fn wide_zero_dividend() {
        let a = BitUInt::zero(200);
        let b = with_bits(200, &[0, 170]);
        assert_words(&(&a / &b), 200, &with_bits(200, &[]).to_words());
        assert_words(&(&a % &b), 200, &with_bits(200, &[]).to_words());
    }

    #[test]
    fn w128_wide_div_wide() {
        let a = with_bits(128, &[0, 1, 2, 63, 127]);
        let b = with_bits(128, &[0, 64]);
        assert_words(&(&a / &b), 128, &with_bits(128, &[63]).to_words());
    }

    #[test]
    fn w128_wide_rem_wide() {
        let a = with_bits(128, &[0, 1, 2, 63, 127]);
        let b = with_bits(128, &[0, 64]);
        assert_words(&(&a % &b), 128, &with_bits(128, &[0, 1, 2]).to_words());
    }

    #[test]
    fn w200_product_above_bit_127() {
        let a = with_bits(200, &[150]);
        let b = with_bits(200, &[20]);
        assert_words(&(&a * &b), 200, &[0, 0, 4_398_046_511_104, 0]);
    }

    #[test]
    fn w200_quotient_above_bit_127() {
        let a = with_bits(200, &[0, 1, 75, 170]);
        let b = with_bits(200, &[65]);
        assert_words(&(&a / &b), 200, &[1024, 2_199_023_255_552, 0, 0]);
    }

    #[test]
    fn w200_remainder_above_bit_127() {
        let a = with_bits(200, &[0, 1, 75, 170]);
        let b = with_bits(200, &[65]);
        assert_words(&(&a % &b), 200, &[3, 0, 0, 0]);
    }

    #[test]
    fn w200_difference_above_bit_127() {
        let a = with_bits(200, &[0, 170]);
        let b = with_bits(200, &[170]);
        assert_words(&(&a - &b), 200, &[1, 0, 0, 0]);
    }

    fn assert_comparison(a: &BitUInt, b: &BitUInt, expected: Ordering) {
        for (left, right, order) in [(a, b, expected), (b, a, expected.reverse())] {
            assert_eq!(left == right, order == Ordering::Equal);
            assert_eq!(left < right, order == Ordering::Less);
            assert_eq!(left > right, order == Ordering::Greater);
            assert_eq!(left.cmp(right), order);
            assert_eq!(left.partial_cmp(right), Some(order));
        }
    }

    #[test]
    fn w128_high_word_vs_zero() {
        assert_comparison(
            &with_bits(128, &[64]),
            &with_bits(128, &[]),
            Ordering::Greater,
        );
    }

    #[test]
    fn w128_high_word_vs_one() {
        assert_comparison(
            &with_bits(128, &[64]),
            &with_bits(128, &[0]),
            Ordering::Greater,
        );
    }

    #[test]
    fn w100_high_bits_differ() {
        assert_comparison(
            &with_bits(100, &[0, 99]),
            &with_bits(100, &[0, 98]),
            Ordering::Greater,
        );
    }

    #[test]
    fn w65_across_word_boundary() {
        assert_comparison(
            &with_bits(65, &[64]),
            &with_bits(65, &(0..64).collect::<Vec<_>>()),
            Ordering::Greater,
        );
    }

    #[test]
    fn cross_width_equal_values() {
        assert_comparison(
            &with_bits(128, &[64]),
            &with_bits(65, &[64]),
            Ordering::Equal,
        );
    }

    #[test]
    fn cross_width_small_vs_large_storage() {
        assert_comparison(
            &with_bits(64, &[0, 2]),
            &with_bits(128, &[0, 2]),
            Ordering::Equal,
        );
    }

    #[test]
    fn w128_equal_wide_values() {
        assert_comparison(
            &with_bits(128, &[0, 2, 100]),
            &with_bits(128, &[0, 2, 100]),
            Ordering::Equal,
        );
    }

    #[test]
    fn mixed_narrow_left_wide_divisor_div() {
        assert_words(&(BitUInt::new(8, 5) / with_bits(65, &[64])), 8, &[0]);
    }

    #[test]
    fn mixed_narrow_left_wide_divisor_rem() {
        assert_words(&(BitUInt::new(8, 5) % with_bits(65, &[64])), 8, &[5]);
    }

    #[test]
    fn mixed_wide_left_narrow_high_bit() {
        assert_words(&(BitUInt::new(128, 3) * with_bits(65, &[64])), 128, &[0, 3]);
    }

    #[test]
    fn from_binary_str_one_then_64_zeros() {
        let value = BitUInt::from_binary_str(&format!("1{}", "0".repeat(64)));
        assert_words(&value, 65, &[0, 1]);
    }

    #[test]
    fn from_binary_str_word_boundaries() {
        for size in [1, 63, 64, 65, 70, 100, 127, 128, 129, 200, 65535] {
            let expected = with_bits(size, &[0, size / 2, size - 1]);
            let binary = expected.to_string();
            assert_words(
                &BitUInt::from_binary_str(&binary),
                size,
                &expected.to_words(),
            );
        }
    }

    #[test]
    fn from_raw_words_short_slice() {
        let value = BitUInt::from_raw_words(100, Box::new([u64::MAX]));
        assert_words(&value, 100, &[u64::MAX, 0]);
        assert!(!value.get_bit(99));
    }

    #[test]
    fn from_raw_words_empty_slice() {
        for size in [1, 8, 64, 65, 100, 128] {
            let value = BitUInt::from_raw_words(size, Box::new([]));
            assert_words(&value, size, &vec![0; usize::from(size).div_ceil(64)]);
            assert!(value.is_zero());
            assert!(!value.get_bit(size - 1));
        }
    }

    #[test]
    fn from_raw_words_excess_words() {
        let words = [u64::MAX; 4];
        assert_words(
            &BitUInt::from_raw_words(70, Box::new(words)),
            70,
            &[u64::MAX, 63],
        );
        assert_words(
            &BitUInt::from_raw_words(128, Box::new(words)),
            128,
            &[u64::MAX, u64::MAX],
        );
        assert_words(&BitUInt::from_raw_words(8, Box::new(words)), 8, &[255]);
    }

    #[test]
    fn from_raw_words_masks_retained_top_word() {
        for words in [vec![7, u64::MAX], vec![7, u64::MAX, 0]] {
            assert_words(
                &BitUInt::from_raw_words(70, words.into_boxed_slice()),
                70,
                &[7, 63],
            );
        }
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn wide_division_by_zero() {
        let _ = with_bits(200, &[170]) / BitUInt::zero(200);
    }

    #[test]
    #[should_panic(expected = "Remainder by zero")]
    fn wide_remainder_by_zero() {
        let _ = with_bits(200, &[170]) % BitUInt::zero(200);
    }

    #[test]
    fn wide_shift_boundaries() {
        // At width 70, shift the bits {0, 63, 64, 69}; discard bits outside 0..70.
        let value = with_bits(70, &[0, 63, 64, 69]);
        let cases: [(u16, &[u16], &[u16]); 7] = [
            (0, &[0, 63, 64, 69], &[0, 63, 64, 69]),
            (63, &[63], &[0, 1, 6]),
            (64, &[64], &[0, 5]),
            (65, &[65], &[4]),
            (69, &[69], &[0]),
            (70, &[], &[]),
            (71, &[], &[]),
        ];
        for (shift, left, right) in cases {
            assert_words(&(&value << shift), 70, &with_bits(70, left).to_words());
            assert_words(&(&value >> shift), 70, &with_bits(70, right).to_words());
        }
    }

    #[test]
    fn normalized_storage_compositions() {
        // Padding preserves all 64 supplied bits; carry/borrow crosses into word one.
        let value = BitUInt::from_raw_words(100, Box::new([u64::MAX]));
        let one = BitUInt::new(8, 1);
        let carried = &value + &one;
        assert_words(&carried, 100, &[0, 1]);
        assert_words(&(&carried - &one), 100, &[u64::MAX, 0]);
        assert_words(&(!&value), 100, &[0, (1 << 36) - 1]);
        assert_words(&(&value << 64), 100, &[0, (1 << 36) - 1]);
        assert_words(&(&carried >> 64), 100, &[1, 0]);
        assert!(!value.get_bit(99));
        let mut high = value.clone();
        high.set_bit(99, true);
        assert!(high.get_bit(99));
        assert_words(&high, 100, &[u64::MAX, 1 << 35]);
        assert_words(&(&value & &high), 100, &[u64::MAX, 0]);
        assert_words(&(&value | &high), 100, &[u64::MAX, 1 << 35]);
        assert_words(&(&value ^ &high), 100, &[0, 1 << 35]);
        let empty = BitUInt::from_raw_words(100, Box::new([]));
        assert_words(&(&empty - &one), 100, &[u64::MAX, (1 << 36) - 1]);
        assert_eq!(value.count_ones(), 64);
        assert_eq!(value.count_zeros(), 36);
        assert_eq!(value.to_u64(), Some(u64::MAX));
        assert_eq!(
            value.to_string(),
            format!("{}{}", "0".repeat(36), "1".repeat(64))
        );
    }

    #[test]
    fn maximum_width() {
        let size = u16::MAX;
        let mut words = vec![0; 1024];
        words[1023] = 1 << 62;
        let high = with_bits(size, &[65534]);
        assert_words(&high, size, &words);
        assert_words(&(&high * &BitUInt::new(1, 1)), size, &words);
        assert_words(&(&high / &high), size, &BitUInt::new(size, 1).to_words());
        assert_words(&(&high % &high), size, &vec![0; 1024]);
        let padded = BitUInt::from_raw_words(size, Box::new([1]));
        assert!(!padded.get_bit(65534));
        assert_words(&(&padded << 65534), size, &words);
    }

    #[test]
    fn constructor_and_bit_index_panics() {
        for constructor in [BitUInt::zero, BitUInt::ones] {
            assert!(std::panic::catch_unwind(|| constructor(0)).is_err());
        }
        assert!(std::panic::catch_unwind(|| BitUInt::from_raw_words(0, Box::new([]))).is_err());
        for input in [
            "",
            "2",
            "+1",
            "1_0",
            "é",
            &"0".repeat(65536),
            &format!("{}2", "0".repeat(64)),
        ] {
            assert!(std::panic::catch_unwind(|| BitUInt::from_binary_str(input)).is_err());
        }
        for size in [8, 100] {
            assert!(std::panic::catch_unwind(|| BitUInt::zero(size).get_bit(size)).is_err());
            assert!(std::panic::catch_unwind(|| BitUInt::zero(size).set_bit(size, true)).is_err());
        }
    }

    #[test]
    fn mixed_width_arithmetic_and_division_identity() {
        for left_width in [1, 8, 64, 65, 70, 128, 200] {
            for right_width in [1, 8, 64, 65, 70, 128, 200] {
                let a = BitUInt::ones(left_width);
                let b = with_bits(right_width, &[0, right_width - 1]);
                let quotient = &a / &b;
                let remainder = &a % &b;
                assert_eq!(quotient.size(), left_width);
                assert_eq!(remainder.size(), left_width);
                assert!(remainder < b);
                // Reconstruct at a width that cannot wrap: a quotient below
                // 2^left_width times a divisor below 2^right_width fits in their
                // sum. Reconstructing at left_width instead would accept wrong
                // pairs, such as q=127, r=0 for a=255, b=129 at width 8.
                let wide = left_width + right_width;
                let widen =
                    |v: &BitUInt| BitUInt::from_raw_words(wide, v.to_words().into_boxed_slice());
                assert_eq!(
                    (&(&widen(&quotient) * &widen(&b)) + &widen(&remainder)).to_words(),
                    widen(&a).to_words()
                );
            }
        }
    }
}
