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

//! A fixed-width integer with explicit bit width tracking.
//!
//! [`BitInt`] provides a runtime-sized integer type that tracks its bit width explicitly.
//! It supports both signed and unsigned semantics, with a fast path for widths ≤64 bits
//! and arbitrary precision for larger widths.
//!
//! # Examples
//!
//! ```
//! use pecos_core::BitInt;
//!
//! // Create an 8-bit unsigned integer
//! let a = BitInt::new_unsigned(8, 0b1010_1010);
//! let b = BitInt::new_unsigned(8, 0b0101_0101);
//!
//! // Bitwise XOR
//! let c = &a ^ &b;
//! assert_eq!(c.to_u64(), Some(0xFF));
//!
//! // Individual bit access
//! assert_eq!(a.get_bit(0), false);
//! assert_eq!(a.get_bit(1), true);
//! ```

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Not, Rem, Shl, Shr, Sub};

/// Internal storage for `BitInt` values.
///
/// Uses a single `u64` for widths ≤64 bits (fast path), and a boxed slice
/// of `u64` words for larger widths (arbitrary precision).
#[derive(Clone, Debug, PartialEq, Eq)]
enum BitIntValue {
    /// Fast path: single 64-bit word for widths ≤64
    Small(u64),
    /// Arbitrary precision: packed u64 words, LSB first
    Large(Box<[u64]>),
}

/// A fixed-width integer with explicit bit width tracking.
///
/// Supports both signed and unsigned semantics:
/// - **Unsigned**: Values are clamped to the bit width after operations
/// - **Signed**: Values can be negative, with sign extension for operations
///
/// The internal representation uses a fast path for ≤64 bits (single `u64`)
/// and falls back to arbitrary precision for larger widths.
#[derive(Clone, Debug)]
pub struct BitInt {
    /// Bit width of this integer (1 to 65535)
    size: u16,
    /// Whether this integer uses signed semantics
    signed: bool,
    /// The actual value storage
    value: BitIntValue,
}

impl BitInt {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a new unsigned `BitInt` with the given size and value.
    ///
    /// The value is clamped to fit within the specified bit width.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    #[must_use]
    pub fn new_unsigned(size: u16, value: u64) -> Self {
        assert!(size > 0, "BitInt size must be at least 1");
        let mut result = Self {
            size,
            signed: false,
            value: if size <= 64 {
                BitIntValue::Small(value)
            } else {
                let num_words = Self::words_needed(size);
                let mut words = vec![0u64; num_words].into_boxed_slice();
                words[0] = value;
                BitIntValue::Large(words)
            },
        };
        result.mask_to_width();
        result
    }

    /// Create a new signed `BitInt` with the given size and value.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    #[must_use]
    pub fn new_signed(size: u16, value: i64) -> Self {
        assert!(size > 0, "BitInt size must be at least 1");
        Self {
            size,
            signed: true,
            value: if size <= 64 {
                // Store as unsigned bits, but track signed semantics
                #[allow(clippy::cast_sign_loss)]
                BitIntValue::Small(value as u64)
            } else {
                let num_words = Self::words_needed(size);
                let mut words = vec![0u64; num_words].into_boxed_slice();
                #[allow(clippy::cast_sign_loss)]
                {
                    words[0] = value as u64;
                }
                // Sign extend if negative
                if value < 0 {
                    for word in words.iter_mut().skip(1) {
                        *word = u64::MAX;
                    }
                }
                BitIntValue::Large(words)
            },
        }
    }

    /// Create a new `BitInt` from a binary string.
    ///
    /// The size is determined by the string length.
    ///
    /// # Panics
    ///
    /// Panics if the string is empty or contains non-binary characters.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // Size is validated below
    pub fn from_binary_str(s: &str) -> Self {
        assert!(!s.is_empty(), "Binary string must not be empty");
        assert!(
            u16::try_from(s.len()).is_ok(),
            "Binary string too long (max 65535 chars)"
        );
        let size = s.len() as u16;

        if size <= 64 {
            let value = u64::from_str_radix(s, 2).expect("Invalid binary string");
            Self::new_unsigned(size, value)
        } else {
            // Parse in 64-bit chunks from the right (LSB first)
            let mut words = Vec::with_capacity(Self::words_needed(size));
            let chars: Vec<char> = s.chars().collect();

            for chunk_start in (0..chars.len()).step_by(64).rev() {
                let chunk_end = chars.len().min(chunk_start + 64);
                let chunk: String = chars[chunk_start..chunk_end].iter().collect();
                let word = u64::from_str_radix(&chunk, 2).expect("Invalid binary string");
                words.push(word);
            }

            // Reverse because we built LSB-first but pushed in wrong order
            words.reverse();

            Self {
                size,
                signed: false,
                value: BitIntValue::Large(words.into_boxed_slice()),
            }
        }
    }

    /// Create a zero value with the given size.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    #[must_use]
    pub fn zero(size: u16, signed: bool) -> Self {
        assert!(size > 0, "BitInt size must be at least 1");
        Self {
            size,
            signed,
            value: if size <= 64 {
                BitIntValue::Small(0)
            } else {
                let num_words = Self::words_needed(size);
                BitIntValue::Large(vec![0u64; num_words].into_boxed_slice())
            },
        }
    }

    /// Create an all-ones value with the given size.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    #[must_use]
    pub fn ones(size: u16, signed: bool) -> Self {
        assert!(size > 0, "BitInt size must be at least 1");
        let mut result = Self {
            size,
            signed,
            value: if size <= 64 {
                BitIntValue::Small(u64::MAX)
            } else {
                let num_words = Self::words_needed(size);
                BitIntValue::Large(vec![u64::MAX; num_words].into_boxed_slice())
            },
        };
        result.mask_to_width();
        result
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Returns the bit width of this integer.
    #[must_use]
    pub fn size(&self) -> u16 {
        self.size
    }

    /// Returns whether this integer uses signed semantics.
    #[must_use]
    pub fn is_signed(&self) -> bool {
        self.signed
    }

    /// Returns the value as a `u64` if it fits, otherwise `None`.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        match &self.value {
            BitIntValue::Small(v) => Some(*v),
            BitIntValue::Large(words) => {
                // Check if all words except the first are zero
                if words.iter().skip(1).all(|&w| w == 0) {
                    Some(words[0])
                } else {
                    None
                }
            }
        }
    }

    /// Returns the value as an `i64` if it fits, otherwise `None`.
    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        match &self.value {
            BitIntValue::Small(v) => {
                if self.signed && self.size < 64 {
                    // Sign extend
                    let sign_bit = 1u64 << (self.size - 1);
                    if *v & sign_bit != 0 {
                        let mask = !((1u64 << self.size) - 1);
                        #[allow(clippy::cast_possible_wrap)]
                        return Some((*v | mask) as i64);
                    }
                }
                #[allow(clippy::cast_possible_wrap)]
                Some(*v as i64)
            }
            BitIntValue::Large(_) => None, // Too large for i64
        }
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
            BitIntValue::Small(v) => (*v >> index) & 1 == 1,
            BitIntValue::Large(words) => {
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
            BitIntValue::Small(v) => {
                if value {
                    *v |= 1 << index;
                } else {
                    *v &= !(1 << index);
                }
            }
            BitIntValue::Large(words) => {
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
            BitIntValue::Small(v) => v.count_ones(),
            BitIntValue::Large(words) => words.iter().map(|w| w.count_ones()).sum(),
        }
    }

    /// Returns the number of 0 bits.
    #[must_use]
    pub fn count_zeros(&self) -> u32 {
        u32::from(self.size) - self.count_ones()
    }

    /// Returns true if the value is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        match &self.value {
            BitIntValue::Small(v) => *v == 0,
            BitIntValue::Large(words) => words.iter().all(|&w| w == 0),
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

    /// Mask the value to fit within the bit width (for unsigned).
    fn mask_to_width(&mut self) {
        if !self.signed {
            match &mut self.value {
                BitIntValue::Small(v) => {
                    if self.size < 64 {
                        *v &= (1u64 << self.size) - 1;
                    }
                }
                BitIntValue::Large(words) => {
                    // Clear bits beyond the size in the last word
                    let last_word_bits = self.size % 64;
                    if last_word_bits > 0 {
                        let last_idx = words.len() - 1;
                        words[last_idx] &= (1u64 << last_word_bits) - 1;
                    }
                }
            }
        }
    }

    /// Get the raw underlying u64 value (for small values or first word of large).
    /// This is used for mixed-size operations that operate on raw values.
    #[must_use]
    fn raw_u64(&self) -> u64 {
        match &self.value {
            BitIntValue::Small(v) => *v,
            BitIntValue::Large(words) => words[0],
        }
    }

    /// Get word at index, or 0 if beyond bounds.
    #[must_use]
    fn word_at(&self, index: usize) -> u64 {
        match &self.value {
            BitIntValue::Small(v) => {
                if index == 0 {
                    *v
                } else {
                    0
                }
            }
            BitIntValue::Large(words) => words.get(index).copied().unwrap_or(0),
        }
    }

    /// Create a new `BitInt` with the same size and signedness, with the given small value.
    fn new_with_same_config(&self, value: u64) -> Self {
        let mut result = Self {
            size: self.size,
            signed: self.signed,
            value: BitIntValue::Small(value),
        };
        if !self.signed {
            result.mask_to_width();
        }
        result
    }

    /// Create a new `BitInt` with the same size and signedness, with large value.
    fn new_with_same_config_large(&self, words: Box<[u64]>) -> Self {
        let mut result = Self {
            size: self.size,
            signed: self.signed,
            value: BitIntValue::Large(words),
        };
        if !self.signed {
            result.mask_to_width();
        }
        result
    }
}

// ============================================================================
// Display
// ============================================================================

impl fmt::Display for BitInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.value {
            BitIntValue::Small(v) => {
                write!(f, "{:0>width$b}", v, width = self.size as usize)
            }
            BitIntValue::Large(_) => {
                // Build binary string from MSB to LSB
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

impl PartialEq for BitInt {
    fn eq(&self, other: &Self) -> bool {
        // Compare raw values directly (like BinArray)
        // Different sizes can still be equal if their values match
        self.raw_u64() == other.raw_u64()
    }
}

impl Eq for BitInt {}

impl PartialOrd for BitInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Compare raw values directly (like BinArray)
        Some(self.cmp(other))
    }
}

impl Ord for BitInt {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_internal(other)
    }
}

impl BitInt {
    fn cmp_internal(&self, other: &Self) -> Ordering {
        // Compare raw values directly (like BinArray)
        // For simplicity, use u64 comparison for most cases
        self.raw_u64().cmp(&other.raw_u64())
    }
}

// ============================================================================
// Bitwise Operations
// ============================================================================

impl BitXor for &BitInt {
    type Output = BitInt;

    fn bitxor(self, rhs: Self) -> BitInt {
        // Operate on raw values, result uses left operand's size (like BinArray)
        match &self.value {
            BitIntValue::Small(_) => self.new_with_same_config(self.raw_u64() ^ rhs.raw_u64()),
            BitIntValue::Large(words) => {
                let result: Box<[u64]> = words
                    .iter()
                    .enumerate()
                    .map(|(i, &w)| w ^ rhs.word_at(i))
                    .collect();
                self.new_with_same_config_large(result)
            }
        }
    }
}

impl BitAnd for &BitInt {
    type Output = BitInt;

    fn bitand(self, rhs: Self) -> BitInt {
        // Operate on raw values, result uses left operand's size (like BinArray)
        match &self.value {
            BitIntValue::Small(_) => self.new_with_same_config(self.raw_u64() & rhs.raw_u64()),
            BitIntValue::Large(words) => {
                let result: Box<[u64]> = words
                    .iter()
                    .enumerate()
                    .map(|(i, &w)| w & rhs.word_at(i))
                    .collect();
                self.new_with_same_config_large(result)
            }
        }
    }
}

impl BitOr for &BitInt {
    type Output = BitInt;

    fn bitor(self, rhs: Self) -> BitInt {
        // Operate on raw values, result uses left operand's size (like BinArray)
        match &self.value {
            BitIntValue::Small(_) => self.new_with_same_config(self.raw_u64() | rhs.raw_u64()),
            BitIntValue::Large(words) => {
                let result: Box<[u64]> = words
                    .iter()
                    .enumerate()
                    .map(|(i, &w)| w | rhs.word_at(i))
                    .collect();
                self.new_with_same_config_large(result)
            }
        }
    }
}

impl Not for &BitInt {
    type Output = BitInt;

    fn not(self) -> BitInt {
        match &self.value {
            BitIntValue::Small(v) => self.new_with_same_config(!v),
            BitIntValue::Large(words) => {
                let new_words: Box<[u64]> = words.iter().map(|w| !w).collect();
                self.new_with_same_config_large(new_words)
            }
        }
    }
}

// ============================================================================
// Shift Operations
// ============================================================================

impl Shl<u16> for &BitInt {
    type Output = BitInt;

    fn shl(self, rhs: u16) -> BitInt {
        if rhs >= self.size {
            return BitInt::zero(self.size, self.signed);
        }

        match &self.value {
            BitIntValue::Small(v) => self.new_with_same_config(v << rhs),
            BitIntValue::Large(words) => {
                let word_shift = (rhs / 64) as usize;
                let bit_shift = rhs % 64;

                let mut new_words = vec![0u64; words.len()];

                for i in word_shift..words.len() {
                    new_words[i] = words[i - word_shift] << bit_shift;
                    if bit_shift > 0 && i > word_shift {
                        new_words[i] |= words[i - word_shift - 1] >> (64 - bit_shift);
                    }
                }

                self.new_with_same_config_large(new_words.into_boxed_slice())
            }
        }
    }
}

impl Shr<u16> for &BitInt {
    type Output = BitInt;

    fn shr(self, rhs: u16) -> BitInt {
        if rhs >= self.size {
            if self.signed && self.get_bit(self.size - 1) {
                // Arithmetic shift: fill with sign bit
                return BitInt::ones(self.size, self.signed);
            }
            return BitInt::zero(self.size, self.signed);
        }

        match &self.value {
            BitIntValue::Small(v) => {
                if self.signed {
                    // Arithmetic shift
                    #[allow(clippy::cast_possible_wrap)]
                    let signed_v = *v as i64;
                    #[allow(clippy::cast_sign_loss)]
                    let shifted = (signed_v >> rhs) as u64;
                    self.new_with_same_config(shifted)
                } else {
                    self.new_with_same_config(v >> rhs)
                }
            }
            BitIntValue::Large(words) => {
                let word_shift = (rhs / 64) as usize;
                let bit_shift = rhs % 64;
                let fill = if self.signed && self.get_bit(self.size - 1) {
                    u64::MAX
                } else {
                    0
                };

                let mut new_words = vec![fill; words.len()];

                for i in 0..(words.len() - word_shift) {
                    new_words[i] = words[i + word_shift] >> bit_shift;
                    if bit_shift > 0 && i + word_shift + 1 < words.len() {
                        new_words[i] |= words[i + word_shift + 1] << (64 - bit_shift);
                    }
                }

                self.new_with_same_config_large(new_words.into_boxed_slice())
            }
        }
    }
}

// ============================================================================
// Arithmetic Operations (Small values only for now)
// ============================================================================

impl Add for &BitInt {
    type Output = BitInt;

    fn add(self, rhs: Self) -> BitInt {
        // Operate on raw values, result uses left operand's size (like BinArray)
        match &self.value {
            BitIntValue::Small(_) => {
                self.new_with_same_config(self.raw_u64().wrapping_add(rhs.raw_u64()))
            }
            BitIntValue::Large(words) => {
                let mut result = vec![0u64; words.len()];
                let mut carry = 0u64;

                for i in 0..words.len() {
                    let (sum1, c1) = words[i].overflowing_add(rhs.word_at(i));
                    let (sum2, c2) = sum1.overflowing_add(carry);
                    result[i] = sum2;
                    carry = u64::from(c1) + u64::from(c2);
                }

                self.new_with_same_config_large(result.into_boxed_slice())
            }
        }
    }
}

impl Sub for &BitInt {
    type Output = BitInt;

    #[allow(clippy::suspicious_arithmetic_impl)] // Using + to accumulate borrows is correct
    fn sub(self, rhs: Self) -> BitInt {
        // Operate on raw values, result uses left operand's size (like BinArray)
        match &self.value {
            BitIntValue::Small(_) => {
                self.new_with_same_config(self.raw_u64().wrapping_sub(rhs.raw_u64()))
            }
            BitIntValue::Large(words) => {
                let mut result = vec![0u64; words.len()];
                let mut borrow = 0u64;

                for i in 0..words.len() {
                    let (diff1, b1) = words[i].overflowing_sub(rhs.word_at(i));
                    let (diff2, b2) = diff1.overflowing_sub(borrow);
                    result[i] = diff2;
                    borrow = u64::from(b1) + u64::from(b2);
                }

                self.new_with_same_config_large(result.into_boxed_slice())
            }
        }
    }
}

impl Mul for &BitInt {
    type Output = BitInt;

    fn mul(self, rhs: Self) -> BitInt {
        // Operate on raw values, result uses left operand's size (like BinArray)
        match &self.value {
            BitIntValue::Small(_) => {
                self.new_with_same_config(self.raw_u64().wrapping_mul(rhs.raw_u64()))
            }
            BitIntValue::Large(_) => {
                // TODO: Implement full large multiplication
                // For now, only support if it fits in u64
                let a = self.raw_u64();
                let b = rhs.raw_u64();
                let mut result = BitInt::zero(self.size, self.signed);
                if let BitIntValue::Large(ref mut words) = result.value {
                    words[0] = a.wrapping_mul(b);
                }
                result.mask_to_width();
                result
            }
        }
    }
}

impl Div for &BitInt {
    type Output = BitInt;

    fn div(self, rhs: Self) -> BitInt {
        // Operate on raw values, result uses left operand's size (like BinArray)
        let a = self.raw_u64();
        let b = rhs.raw_u64();
        assert!(b != 0, "Division by zero");

        match &self.value {
            BitIntValue::Small(_) => {
                if self.signed {
                    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
                    let result = (a as i64 / b as i64) as u64;
                    self.new_with_same_config(result)
                } else {
                    self.new_with_same_config(a / b)
                }
            }
            BitIntValue::Large(_) => {
                let mut result = BitInt::zero(self.size, self.signed);
                if let BitIntValue::Large(ref mut words) = result.value {
                    words[0] = a / b;
                }
                result
            }
        }
    }
}

impl Rem for &BitInt {
    type Output = BitInt;

    fn rem(self, rhs: Self) -> BitInt {
        // Operate on raw values, result uses left operand's size (like BinArray)
        let a = self.raw_u64();
        let b = rhs.raw_u64();
        assert!(b != 0, "Remainder by zero");

        match &self.value {
            BitIntValue::Small(_) => {
                if self.signed {
                    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
                    let result = (a as i64 % b as i64) as u64;
                    self.new_with_same_config(result)
                } else {
                    self.new_with_same_config(a % b)
                }
            }
            BitIntValue::Large(_) => {
                let mut result = BitInt::zero(self.size, self.signed);
                if let BitIntValue::Large(ref mut words) = result.value {
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
        impl $trait for BitInt {
            type Output = BitInt;
            fn $method(self, rhs: Self) -> BitInt {
                (&self).$method(&rhs)
            }
        }
        impl $trait<&BitInt> for BitInt {
            type Output = BitInt;
            fn $method(self, rhs: &BitInt) -> BitInt {
                (&self).$method(rhs)
            }
        }
        impl $trait<BitInt> for &BitInt {
            type Output = BitInt;
            fn $method(self, rhs: BitInt) -> BitInt {
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

impl Not for BitInt {
    type Output = BitInt;
    fn not(self) -> BitInt {
        (&self).not()
    }
}

impl Shl<u16> for BitInt {
    type Output = BitInt;
    fn shl(self, rhs: u16) -> BitInt {
        (&self).shl(rhs)
    }
}

impl Shr<u16> for BitInt {
    type Output = BitInt;
    fn shr(self, rhs: u16) -> BitInt {
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
    fn test_new_unsigned() {
        let a = BitInt::new_unsigned(8, 0xFF);
        assert_eq!(a.size(), 8);
        assert!(!a.is_signed());
        assert_eq!(a.to_u64(), Some(0xFF));

        // Test clamping
        let b = BitInt::new_unsigned(4, 0xFF);
        assert_eq!(b.to_u64(), Some(0x0F));
    }

    #[test]
    fn test_new_signed() {
        let a = BitInt::new_signed(8, -1);
        assert_eq!(a.size(), 8);
        assert!(a.is_signed());
        assert_eq!(a.to_i64(), Some(-1));
    }

    #[test]
    fn test_from_binary_str() {
        let a = BitInt::from_binary_str("1010");
        assert_eq!(a.size(), 4);
        assert_eq!(a.to_u64(), Some(0b1010));
    }

    #[test]
    fn test_display() {
        let a = BitInt::new_unsigned(8, 0b1010_0101);
        assert_eq!(format!("{a}"), "10100101");

        let b = BitInt::new_unsigned(4, 0b0101);
        assert_eq!(format!("{b}"), "0101");
    }

    #[test]
    fn test_bit_access() {
        let mut a = BitInt::new_unsigned(8, 0b1010_0101);
        assert!(a.get_bit(0));
        assert!(!a.get_bit(1));
        assert!(a.get_bit(2));

        a.set_bit(1, true);
        assert!(a.get_bit(1));
        assert_eq!(a.to_u64(), Some(0b1010_0111));
    }

    #[test]
    fn test_bitwise_xor() {
        let a = BitInt::new_unsigned(8, 0b1010_1010);
        let b = BitInt::new_unsigned(8, 0b0101_0101);
        let c = &a ^ &b;
        assert_eq!(c.to_u64(), Some(0xFF));
    }

    #[test]
    fn test_bitwise_and() {
        let a = BitInt::new_unsigned(8, 0b1010_1010);
        let b = BitInt::new_unsigned(8, 0b1111_0000);
        let c = &a & &b;
        assert_eq!(c.to_u64(), Some(0b1010_0000));
    }

    #[test]
    fn test_bitwise_or() {
        let a = BitInt::new_unsigned(8, 0b1010_0000);
        let b = BitInt::new_unsigned(8, 0b0000_0101);
        let c = &a | &b;
        assert_eq!(c.to_u64(), Some(0b1010_0101));
    }

    #[test]
    fn test_bitwise_not() {
        let a = BitInt::new_unsigned(8, 0b1010_1010);
        let b = !&a;
        assert_eq!(b.to_u64(), Some(0b0101_0101));
    }

    #[test]
    fn test_shift_left() {
        let a = BitInt::new_unsigned(8, 0b0000_1111);
        let b = &a << 4;
        assert_eq!(b.to_u64(), Some(0b1111_0000));
    }

    #[test]
    fn test_shift_right() {
        let a = BitInt::new_unsigned(8, 0b1111_0000);
        let b = &a >> 4;
        assert_eq!(b.to_u64(), Some(0b0000_1111));
    }

    #[test]
    fn test_arithmetic_add() {
        let left = BitInt::new_unsigned(8, 100);
        let right = BitInt::new_unsigned(8, 50);
        let sum = &left + &right;
        assert_eq!(sum.to_u64(), Some(150));

        // Test overflow wrapping
        let large_left = BitInt::new_unsigned(8, 200);
        let large_right = BitInt::new_unsigned(8, 100);
        let overflow_sum = &large_left + &large_right;
        assert_eq!(overflow_sum.to_u64(), Some(44)); // (200 + 100) % 256 = 44
    }

    #[test]
    fn test_arithmetic_sub() {
        let a = BitInt::new_unsigned(8, 100);
        let b = BitInt::new_unsigned(8, 50);
        let c = &a - &b;
        assert_eq!(c.to_u64(), Some(50));
    }

    #[test]
    fn test_arithmetic_mul() {
        let a = BitInt::new_unsigned(8, 10);
        let b = BitInt::new_unsigned(8, 5);
        let c = &a * &b;
        assert_eq!(c.to_u64(), Some(50));
    }

    #[test]
    fn test_arithmetic_div() {
        let a = BitInt::new_unsigned(8, 100);
        let b = BitInt::new_unsigned(8, 10);
        let c = &a / &b;
        assert_eq!(c.to_u64(), Some(10));
    }

    #[test]
    fn test_arithmetic_rem() {
        let a = BitInt::new_unsigned(8, 100);
        let b = BitInt::new_unsigned(8, 30);
        let c = &a % &b;
        assert_eq!(c.to_u64(), Some(10));
    }

    #[test]
    fn test_comparison() {
        let a = BitInt::new_unsigned(8, 100);
        let b = BitInt::new_unsigned(8, 50);
        let c = BitInt::new_unsigned(8, 100);

        assert!(a > b);
        assert!(b < a);
        assert_eq!(a, c);
    }

    #[test]
    fn test_count_ones() {
        let a = BitInt::new_unsigned(8, 0b1010_1010);
        assert_eq!(a.count_ones(), 4);
    }

    #[test]
    fn test_large_bitint() {
        let a = BitInt::new_unsigned(128, 0xFFFF_FFFF_FFFF_FFFF);
        assert_eq!(a.size(), 128);
        assert_eq!(a.to_u64(), Some(0xFFFF_FFFF_FFFF_FFFF));

        // Test bit access in large value
        assert!(a.get_bit(0));
        assert!(a.get_bit(63));
        assert!(!a.get_bit(64)); // Second word should be 0
    }

    // Mixed-size operation tests (BinArray-compatible behavior)

    #[test]
    fn test_mixed_size_xor() {
        // 8-bit XOR with 4-bit, result should be 8-bit with left's size
        let a = BitInt::new_unsigned(8, 0b1010_1010);
        let b = BitInt::new_unsigned(4, 0b0101); // Only 4 bits: 0101
        let c = &a ^ &b;
        assert_eq!(c.size(), 8); // Result uses left operand's size
        assert_eq!(c.to_u64(), Some(0b1010_1111)); // XOR with 0101 at lower bits
    }

    #[test]
    fn test_mixed_size_and() {
        let a = BitInt::new_unsigned(8, 0b1111_1111);
        let b = BitInt::new_unsigned(4, 0b1010);
        let c = &a & &b;
        assert_eq!(c.size(), 8);
        assert_eq!(c.to_u64(), Some(0b0000_1010)); // Only lower 4 bits match
    }

    #[test]
    fn test_mixed_size_or() {
        let a = BitInt::new_unsigned(8, 0b1111_0000);
        let b = BitInt::new_unsigned(4, 0b0101);
        let c = &a | &b;
        assert_eq!(c.size(), 8);
        assert_eq!(c.to_u64(), Some(0b1111_0101));
    }

    #[test]
    fn test_mixed_size_add() {
        let a = BitInt::new_unsigned(8, 200);
        let b = BitInt::new_unsigned(4, 10);
        let c = &a + &b;
        assert_eq!(c.size(), 8);
        assert_eq!(c.to_u64(), Some(210));
    }

    #[test]
    fn test_mixed_size_sub() {
        let a = BitInt::new_unsigned(8, 200);
        let b = BitInt::new_unsigned(4, 10);
        let c = &a - &b;
        assert_eq!(c.size(), 8);
        assert_eq!(c.to_u64(), Some(190));
    }

    #[test]
    fn test_mixed_size_comparison() {
        // Different sizes, same value
        let a = BitInt::new_unsigned(8, 10);
        let b = BitInt::new_unsigned(4, 10);
        assert_eq!(a, b); // Same underlying value, should be equal

        let c = BitInt::new_unsigned(8, 20);
        let d = BitInt::new_unsigned(4, 10);
        assert!(c > d);
        assert!(d < c);
    }
}
