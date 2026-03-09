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
                BitUIntValue::Small(words[0])
            } else {
                BitUIntValue::Large(words)
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
    /// Panics if the string is empty or contains non-binary characters.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_binary_str(s: &str) -> Self {
        assert!(!s.is_empty(), "Binary string must not be empty");
        assert!(
            u16::try_from(s.len()).is_ok(),
            "Binary string too long (max 65535 chars)"
        );
        let size = s.len() as u16;

        if size <= 64 {
            let value = u64::from_str_radix(s, 2).expect("Invalid binary string");
            Self::new(size, value)
        } else {
            let mut words = Vec::with_capacity(Self::words_needed(size));
            let chars: Vec<char> = s.chars().collect();

            for chunk_start in (0..chars.len()).step_by(64).rev() {
                let chunk_end = chars.len().min(chunk_start + 64);
                let chunk: String = chars[chunk_start..chunk_end].iter().collect();
                let word = u64::from_str_radix(&chunk, 2).expect("Invalid binary string");
                words.push(word);
            }

            words.reverse();

            Self {
                size,
                value: BitUIntValue::Large(words.into_boxed_slice()),
            }
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
        let mut result = Self {
            size: self.size,
            value: BitUIntValue::Small(value),
        };
        result.mask_to_width();
        result
    }

    /// Create a new `BitUInt` with the same size as self, with large value.
    fn new_with_same_size_large(&self, words: Box<[u64]>) -> Self {
        let mut result = Self {
            size: self.size,
            value: BitUIntValue::Large(words),
        };
        result.mask_to_width();
        result
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
        self.raw_u64() == other.raw_u64()
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
        self.raw_u64().cmp(&other.raw_u64())
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
            BitUIntValue::Large(_) => {
                let a = self.raw_u64();
                let b = rhs.raw_u64();
                let mut result = BitUInt::zero(self.size);
                if let BitUIntValue::Large(ref mut words) = result.value {
                    words[0] = a.wrapping_mul(b);
                }
                result.mask_to_width();
                result
            }
        }
    }
}

impl Div for &BitUInt {
    type Output = BitUInt;

    /// Always unsigned division.
    fn div(self, rhs: Self) -> BitUInt {
        let a = self.raw_u64();
        let b = rhs.raw_u64();
        assert!(b != 0, "Division by zero");

        match &self.value {
            BitUIntValue::Small(_) => self.new_with_same_size(a / b),
            BitUIntValue::Large(_) => {
                let mut result = BitUInt::zero(self.size);
                if let BitUIntValue::Large(ref mut words) = result.value {
                    words[0] = a / b;
                }
                result
            }
        }
    }
}

impl Rem for &BitUInt {
    type Output = BitUInt;

    /// Always unsigned remainder.
    fn rem(self, rhs: Self) -> BitUInt {
        let a = self.raw_u64();
        let b = rhs.raw_u64();
        assert!(b != 0, "Remainder by zero");

        match &self.value {
            BitUIntValue::Small(_) => self.new_with_same_size(a % b),
            BitUIntValue::Large(_) => {
                let mut result = BitUInt::zero(self.size);
                if let BitUIntValue::Large(ref mut words) = result.value {
                    words[0] = a % b;
                }
                result
            }
        }
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
}
