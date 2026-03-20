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

//! A fixed-width signed integer that wraps [`BitUInt`].
//!
//! [`BitInt`] wraps `BitUInt(N+1)` where the extra bit is the sign bit.
//! This means `BitInt(1, 1)` returns 1 (not -1), because the value is stored
//! as `BitUInt(2)` = `0b01` with sign bit 0.
//!
//! # Examples
//!
//! ```
//! use pecos_core::BitInt;
//!
//! let a = BitInt::new(8, 42);
//! assert_eq!(a.to_i64(), Some(42));
//!
//! let b = BitInt::new(8, -1);
//! assert_eq!(b.to_i64(), Some(-1));
//!
//! // 1-bit signed: value 1 is positive (not -1)
//! let c = BitInt::new(1, 1);
//! assert_eq!(c.to_i64(), Some(1));
//!
//! // 1-bit signed: value -1 is negative
//! let d = BitInt::new(1, -1);
//! assert_eq!(d.to_i64(), Some(-1));
//! ```

use crate::bit_uint::BitUInt;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Not, Rem, Shl, Shr, Sub};

/// A fixed-width signed integer that wraps `BitUInt(N+1)`.
///
/// The user-visible size is N bits, but internally N+1 bits are stored
/// where bit N is the sign bit. This allows `BitInt(1, 1)` to be positive.
#[derive(Clone, Debug)]
pub struct BitInt {
    /// User-declared bit width (1 to 65534)
    user_size: u16,
    /// Internal storage: `BitUInt(user_size` + 1), bit `user_size` is the sign bit
    inner: BitUInt,
}

impl BitInt {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a new signed `BitInt` with the given size and value.
    ///
    /// Internally stores the value in `BitUInt(size+1)` using two's complement.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0 or greater than 65534.
    #[must_use]
    pub fn new(size: u16, value: i64) -> Self {
        assert!(size > 0, "BitInt size must be at least 1");
        assert!(size <= 65534, "BitInt size must be at most 65534");
        let internal_size = size + 1;

        #[allow(clippy::cast_sign_loss)]
        let raw = value as u64;

        let inner = if internal_size <= 64 {
            BitUInt::new(internal_size, raw)
        } else {
            // Sign-extend for negative values into upper words
            let num_words = (internal_size as usize).div_ceil(64);
            let mut words = vec![0u64; num_words];
            words[0] = raw;
            if value < 0 {
                for word in words.iter_mut().skip(1) {
                    *word = u64::MAX;
                }
            }
            BitUInt::from_raw_words(internal_size, words.into_boxed_slice())
        };

        Self {
            user_size: size,
            inner,
        }
    }

    /// Create a `BitInt` from a u64 value. The value goes in the lower N bits,
    /// sign bit is always 0 (positive).
    ///
    /// Used for binary string construction and when the raw bit pattern is unsigned.
    #[must_use]
    pub fn new_from_u64(size: u16, value: u64) -> Self {
        assert!(size > 0, "BitInt size must be at least 1");
        assert!(size <= 65534, "BitInt size must be at most 65534");
        let internal_size = size + 1;

        // Mask value to user_size bits to ensure sign bit is 0
        let masked = if size < 64 {
            value & ((1u64 << size) - 1)
        } else {
            value
        };

        Self {
            user_size: size,
            inner: BitUInt::new(internal_size, masked),
        }
    }

    /// Create a `BitInt` from raw inner words representing the `BitUInt(size+1)` value.
    ///
    /// The words represent the internal two's complement value in little-endian order.
    /// The value is masked to fit within `size+1` bits by `BitUInt::from_raw_words`.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0 or greater than 65534.
    #[must_use]
    pub fn new_from_raw_inner(size: u16, inner_words: Box<[u64]>) -> Self {
        assert!(size > 0, "BitInt size must be at least 1");
        assert!(size <= 65534, "BitInt size must be at most 65534");
        let internal_size = size + 1;
        Self {
            user_size: size,
            inner: BitUInt::from_raw_words(internal_size, inner_words),
        }
    }

    /// Create a `BitInt` from a binary string.
    ///
    /// The size is determined by the string length. The sign bit is implicitly 0.
    ///
    /// # Panics
    ///
    /// Panics if the string is empty or contains non-binary characters.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_binary_str(s: &str) -> Self {
        assert!(!s.is_empty(), "Binary string must not be empty");
        assert!(s.len() <= 65534, "Binary string too long (max 65534 chars)");
        let user_size = s.len() as u16;

        // Parse the value from the binary string
        let val = if user_size <= 64 {
            u64::from_str_radix(s, 2).expect("Invalid binary string")
        } else {
            // For large values, just use 0 for now (simplified)
            0
        };

        Self::new_from_u64(user_size, val)
    }

    /// Create a zero value with the given size.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0 or greater than 65534.
    #[must_use]
    pub fn zero(size: u16) -> Self {
        assert!(size > 0, "BitInt size must be at least 1");
        assert!(size <= 65534, "BitInt size must be at most 65534");
        Self {
            user_size: size,
            inner: BitUInt::zero(size + 1),
        }
    }

    /// Create an all-ones value (all N data bits set, sign bit 0 = max positive).
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0 or greater than 65534.
    #[must_use]
    pub fn ones(size: u16) -> Self {
        assert!(size > 0, "BitInt size must be at least 1");
        assert!(size <= 65534, "BitInt size must be at most 65534");
        let internal_size = size + 1;

        // All N data bits set, sign bit 0
        // Value = (1 << size) - 1
        let val = if size < 64 {
            (1u64 << size) - 1
        } else {
            u64::MAX
        };

        Self {
            user_size: size,
            inner: BitUInt::new(internal_size, val),
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Returns the user-declared bit width (not the internal size).
    #[must_use]
    pub fn size(&self) -> u16 {
        self.user_size
    }

    /// Always returns true (signed).
    #[must_use]
    pub fn is_signed(&self) -> bool {
        true
    }

    /// Returns the value as an `i64` by sign-extending from N+1-bit two's complement.
    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        let internal_size = self.inner.size(); // = user_size + 1
        if internal_size > 64 {
            return None;
        }
        let raw = self.inner.raw_u64();

        if internal_size == 64 {
            #[allow(clippy::cast_possible_wrap)]
            return Some(raw as i64);
        }

        // internal_size < 64: sign extend from bit (internal_size - 1)
        let sign_bit = 1u64 << (internal_size - 1);
        if raw & sign_bit != 0 {
            let mask = !((1u64 << internal_size) - 1);
            #[allow(clippy::cast_possible_wrap)]
            Some((raw | mask) as i64)
        } else {
            #[allow(clippy::cast_possible_wrap)]
            Some(raw as i64)
        }
    }

    /// Returns the raw N+1-bit unsigned value.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        self.inner.to_u64()
    }

    /// Get the value of a specific bit (0-indexed from LSB).
    /// Bounds-checked against `user_size` (cannot access the sign bit by index).
    ///
    /// # Panics
    ///
    /// Panics if `index >= user_size`.
    #[must_use]
    pub fn get_bit(&self, index: u16) -> bool {
        assert!(index < self.user_size, "Bit index out of bounds");
        self.inner.get_bit(index)
    }

    /// Set the value of a specific bit (0-indexed from LSB).
    /// Bounds-checked against `user_size` (cannot access the sign bit by index).
    ///
    /// # Panics
    ///
    /// Panics if `index >= user_size`.
    pub fn set_bit(&mut self, index: u16, value: bool) {
        assert!(index < self.user_size, "Bit index out of bounds");
        self.inner.set_bit(index, value);
    }

    /// Returns the number of 1 bits in the user data bits (excludes sign bit).
    #[must_use]
    pub fn count_ones(&self) -> u32 {
        let total = self.inner.count_ones();
        // Subtract the sign bit if it's set
        if self.inner.get_bit(self.user_size) {
            total - 1
        } else {
            total
        }
    }

    /// Returns the number of 0 bits in the user data bits.
    #[must_use]
    pub fn count_zeros(&self) -> u32 {
        u32::from(self.user_size) - self.count_ones()
    }

    /// Returns true if the value is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Returns true if the value is negative (sign bit is set).
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.inner.get_bit(self.user_size)
    }

    /// Returns a reference to the inner `BitUInt`.
    #[must_use]
    pub fn inner(&self) -> &BitUInt {
        &self.inner
    }

    /// Returns the internal `BitUInt(size+1)` value as u64 words (little-endian, LSB first).
    #[must_use]
    pub fn inner_words(&self) -> Vec<u64> {
        self.inner.to_words()
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Create a new `BitInt` with the same `user_size`, wrapping the given inner `BitUInt`.
    fn wrap_result(&self, inner: BitUInt) -> Self {
        Self {
            user_size: self.user_size,
            inner,
        }
    }
}

// ============================================================================
// Display (shows only the N user bits, not the sign bit)
// ============================================================================

impl fmt::Display for BitInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::with_capacity(self.user_size as usize);
        for i in (0..self.user_size).rev() {
            s.push(if self.inner.get_bit(i) { '1' } else { '0' });
        }
        write!(f, "{s}")
    }
}

// ============================================================================
// Equality and Ordering (signed comparison)
// ============================================================================

impl PartialEq for BitInt {
    fn eq(&self, other: &Self) -> bool {
        self.inner.raw_u64() == other.inner.raw_u64()
    }
}

impl Eq for BitInt {}

impl PartialOrd for BitInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BitInt {
    fn cmp(&self, other: &Self) -> Ordering {
        // Signed comparison using the N+1-bit two's complement values
        let self_neg = self.is_negative();
        let other_neg = other.is_negative();

        match (self_neg, other_neg) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => {
                // Same sign: unsigned comparison of raw values is correct
                self.inner.raw_u64().cmp(&other.inner.raw_u64())
            }
        }
    }
}

// ============================================================================
// Bitwise Operations (delegate to inner BitUInt)
// ============================================================================

impl BitXor for &BitInt {
    type Output = BitInt;

    fn bitxor(self, rhs: Self) -> BitInt {
        self.wrap_result(&self.inner ^ &rhs.inner)
    }
}

impl BitAnd for &BitInt {
    type Output = BitInt;

    fn bitand(self, rhs: Self) -> BitInt {
        self.wrap_result(&self.inner & &rhs.inner)
    }
}

impl BitOr for &BitInt {
    type Output = BitInt;

    fn bitor(self, rhs: Self) -> BitInt {
        self.wrap_result(&self.inner | &rhs.inner)
    }
}

impl Not for &BitInt {
    type Output = BitInt;

    fn not(self) -> BitInt {
        self.wrap_result(!&self.inner)
    }
}

// ============================================================================
// Shift Operations
// ============================================================================

impl Shl<u16> for &BitInt {
    type Output = BitInt;

    fn shl(self, rhs: u16) -> BitInt {
        self.wrap_result(&self.inner << rhs)
    }
}

impl Shr<u16> for &BitInt {
    type Output = BitInt;

    /// Arithmetic shift right: fills with sign bit.
    fn shr(self, rhs: u16) -> BitInt {
        let internal_size = self.inner.size();

        if rhs >= internal_size {
            if self.is_negative() {
                return self.wrap_result(BitUInt::ones(internal_size));
            }
            return self.wrap_result(BitUInt::zero(internal_size));
        }

        // Logical shift the inner BitUInt
        let shifted = &self.inner >> rhs;

        if self.is_negative() {
            // Fill the top `rhs` bits with 1s (arithmetic shift)
            let mut result = shifted;
            let start = internal_size.saturating_sub(rhs);
            for i in start..internal_size {
                result.set_bit(i, true);
            }
            self.wrap_result(result)
        } else {
            self.wrap_result(shifted)
        }
    }
}

// ============================================================================
// Arithmetic Operations (delegate to inner, re-wrap)
// ============================================================================

impl Add for &BitInt {
    type Output = BitInt;

    fn add(self, rhs: Self) -> BitInt {
        self.wrap_result(&self.inner + &rhs.inner)
    }
}

impl Sub for &BitInt {
    type Output = BitInt;

    fn sub(self, rhs: Self) -> BitInt {
        self.wrap_result(&self.inner - &rhs.inner)
    }
}

impl Mul for &BitInt {
    type Output = BitInt;

    fn mul(self, rhs: Self) -> BitInt {
        self.wrap_result(&self.inner * &rhs.inner)
    }
}

impl Div for &BitInt {
    type Output = BitInt;

    /// Signed division.
    fn div(self, rhs: Self) -> BitInt {
        let a = self.to_i64().expect("BitInt too large for division");
        let b = rhs.to_i64().expect("BitInt too large for division");
        assert!(b != 0, "Division by zero");

        #[allow(clippy::cast_sign_loss)]
        let result = (a / b) as u64;
        let internal_size = self.inner.size();
        self.wrap_result(BitUInt::new(internal_size, result))
    }
}

impl Rem for &BitInt {
    type Output = BitInt;

    /// Signed remainder.
    fn rem(self, rhs: Self) -> BitInt {
        let a = self.to_i64().expect("BitInt too large for remainder");
        let b = rhs.to_i64().expect("BitInt too large for remainder");
        assert!(b != 0, "Remainder by zero");

        #[allow(clippy::cast_sign_loss)]
        let result = (a % b) as u64;
        let internal_size = self.inner.size();
        self.wrap_result(BitUInt::new(internal_size, result))
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
    fn test_new_positive() {
        let a = BitInt::new(8, 42);
        assert_eq!(a.size(), 8);
        assert!(a.is_signed());
        assert_eq!(a.to_i64(), Some(42));
    }

    #[test]
    fn test_new_negative() {
        let a = BitInt::new(8, -1);
        assert_eq!(a.size(), 8);
        assert_eq!(a.to_i64(), Some(-1));
    }

    #[test]
    fn test_1bit_positive() {
        // The core motivation: BitInt(1, 1) returns 1, not -1
        let a = BitInt::new(1, 1);
        assert_eq!(a.to_i64(), Some(1));
    }

    #[test]
    fn test_1bit_negative() {
        let a = BitInt::new(1, -1);
        assert_eq!(a.to_i64(), Some(-1));
    }

    #[test]
    fn test_1bit_zero() {
        let a = BitInt::new(1, 0);
        assert_eq!(a.to_i64(), Some(0));
        assert!(a.is_zero());
    }

    #[test]
    fn test_from_binary_str() {
        let a = BitInt::from_binary_str("1010");
        assert_eq!(a.size(), 4);
        assert_eq!(a.to_i64(), Some(10));
    }

    #[test]
    fn test_new_from_u64() {
        let a = BitInt::new_from_u64(4, 0b1010);
        assert_eq!(a.to_i64(), Some(10));
    }

    #[test]
    fn test_zero() {
        let a = BitInt::zero(8);
        assert_eq!(a.to_i64(), Some(0));
        assert!(a.is_zero());
    }

    #[test]
    fn test_ones_max_positive() {
        let a = BitInt::ones(8);
        assert_eq!(a.to_i64(), Some(255)); // All 8 data bits set, sign bit 0
    }

    #[test]
    fn test_display() {
        let a = BitInt::new(8, 42);
        // 42 = 0b00101010, display shows 8 user bits
        assert_eq!(format!("{a}"), "00101010");

        let b = BitInt::from_binary_str("1010");
        assert_eq!(format!("{b}"), "1010");
    }

    #[test]
    fn test_bit_access() {
        let mut a = BitInt::new(8, 0b1010_0101);
        assert!(a.get_bit(0));
        assert!(!a.get_bit(1));
        assert!(a.get_bit(2));

        a.set_bit(1, true);
        assert!(a.get_bit(1));
    }

    #[test]
    #[should_panic(expected = "Bit index out of bounds")]
    fn test_bit_access_sign_bit_blocked() {
        let a = BitInt::new(4, 5);
        let _ = a.get_bit(4); // Should panic: can't access sign bit
    }

    #[test]
    fn test_add() {
        let a = BitInt::new(8, 100);
        let b = BitInt::new(8, 50);
        let c = &a + &b;
        assert_eq!(c.to_i64(), Some(150));
    }

    #[test]
    fn test_add_negative() {
        let a = BitInt::new(8, 5);
        let b = BitInt::new(8, -3);
        let c = &a + &b;
        assert_eq!(c.to_i64(), Some(2));
    }

    #[test]
    fn test_sub() {
        let a = BitInt::new(8, 100);
        let b = BitInt::new(8, 50);
        let c = &a - &b;
        assert_eq!(c.to_i64(), Some(50));
    }

    #[test]
    fn test_mul() {
        let a = BitInt::new(8, 10);
        let b = BitInt::new(8, 5);
        let c = &a * &b;
        assert_eq!(c.to_i64(), Some(50));
    }

    #[test]
    fn test_mul_negative() {
        let a = BitInt::new(8, 10);
        let b = BitInt::new(8, -5);
        let c = &a * &b;
        assert_eq!(c.to_i64(), Some(-50));
    }

    #[test]
    fn test_div_signed() {
        let a = BitInt::new(8, -100);
        let b = BitInt::new(8, 10);
        let c = &a / &b;
        assert_eq!(c.to_i64(), Some(-10));
    }

    #[test]
    fn test_rem_signed() {
        let a = BitInt::new(8, -7);
        let b = BitInt::new(8, 3);
        let c = &a % &b;
        assert_eq!(c.to_i64(), Some(-1));
    }

    #[test]
    fn test_signed_comparison() {
        let pos = BitInt::new(8, 5);
        let neg = BitInt::new(8, -5);
        let zero = BitInt::new(8, 0);

        assert!(neg < zero);
        assert!(neg < pos);
        assert!(zero < pos);
        assert!(pos > neg);
    }

    #[test]
    fn test_bitwise_xor() {
        let a = BitInt::new(8, 0b1010_1010);
        let b = BitInt::new(8, 0b0101_0101);
        let c = &a ^ &b;
        // XOR of the inner BitUInts
        let val = c.to_u64().unwrap();
        // Both values fit in 8 bits with sign bit 0 in 9-bit internal
        // a inner: 0b0_1010_1010, b inner: 0b0_0101_0101
        // XOR: 0b0_1111_1111
        assert_eq!(val, 0xFF);
    }

    #[test]
    fn test_bitwise_not() {
        let a = BitInt::new(8, 5);
        let b = !&a;
        assert_eq!(b.to_i64(), Some(-6)); // ~5 = -6 in signed
    }

    #[test]
    fn test_shift_left() {
        let a = BitInt::new(8, 0b0000_1111);
        let b = &a << 4;
        assert_eq!(b.to_i64(), Some(0b1111_0000));
    }

    #[test]
    fn test_shift_right_arithmetic() {
        // Negative value: arithmetic shift fills with 1s
        let a = BitInt::new(8, -8); // 0b11111000 in 8-bit
        let b = &a >> 2;
        assert_eq!(b.to_i64(), Some(-2)); // -8 >> 2 = -2 (arithmetic)
    }

    #[test]
    fn test_shift_right_positive() {
        let a = BitInt::new(8, 8);
        let b = &a >> 2;
        assert_eq!(b.to_i64(), Some(2));
    }

    #[test]
    fn test_is_negative() {
        assert!(BitInt::new(8, -1).is_negative());
        assert!(!BitInt::new(8, 0).is_negative());
        assert!(!BitInt::new(8, 1).is_negative());
    }

    #[test]
    fn test_count_ones() {
        let a = BitInt::new(8, 0b1010_1010);
        assert_eq!(a.count_ones(), 4); // excludes sign bit
    }

    #[test]
    fn test_count_zeros() {
        let a = BitInt::new(8, 0b1010_1010);
        assert_eq!(a.count_zeros(), 4); // excludes sign bit
    }

    #[test]
    fn test_count_ones_negative() {
        let a = BitInt::new(8, -1);
        // -1 in 9-bit two's complement: sign bit 1, data bits all 1
        // count_ones should count all 8 user bits
        assert_eq!(a.count_ones(), 8);
    }

    #[test]
    fn test_shift_right_negative_one() {
        // -1 >> n should still be -1 for arithmetic shift
        let a = BitInt::new(8, -1);
        let b = &a >> 4;
        assert_eq!(b.to_i64(), Some(-1));
    }

    #[test]
    fn test_sub_negative() {
        let a = BitInt::new(8, 5);
        let b = BitInt::new(8, 10);
        let c = &a - &b;
        assert_eq!(c.to_i64(), Some(-5));
    }

    #[test]
    fn test_display_negative() {
        let a = BitInt::new(8, -1);
        let s = format!("{a}");
        assert_eq!(s.len(), 8); // shows 8 user bits
        assert_eq!(s, "11111111"); // all user bits set for -1
    }
}
