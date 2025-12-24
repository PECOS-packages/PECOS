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

//! A single bit type for measurement results.
//!
//! [`Bit`] is a newtype wrapper around `bool` that:
//! - Displays as `0` or `1` instead of `true` or `false`
//! - Supports all bitwise operations (`^`, `&`, `|`, `!`)
//! - Can be used in `if` conditions via `Deref` to `bool`
//! - Converts seamlessly to/from `bool` and integer types
//!
//! # Example
//!
//! ```
//! use pecos_core::Bit;
//!
//! let a = Bit::ONE;
//! let b = Bit::ZERO;
//!
//! // Displays as 0/1
//! assert_eq!(format!("{}", a), "1");
//! assert_eq!(format!("{}", b), "0");
//!
//! // Bitwise operations
//! assert_eq!(a ^ b, Bit::ONE);
//! assert_eq!(a & b, Bit::ZERO);
//! assert_eq!(!b, Bit::ONE);
//!
//! // Use in conditions (via Deref)
//! if *a {
//!     println!("a is one");
//! }
//!
//! // Convert from bool
//! let c = Bit::from(true);
//! assert_eq!(c, Bit::ONE);
//! ```

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Deref, Not};

/// A single bit representing a measurement outcome.
///
/// This type wraps a `bool` but displays as `0` or `1`, which is more natural
/// for quantum measurement results. It implements all the standard bitwise
/// operations and can be used anywhere a `bool` would be used.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Bit(pub bool);

impl Bit {
    /// The zero bit (false).
    pub const ZERO: Bit = Bit(false);

    /// The one bit (true).
    pub const ONE: Bit = Bit(true);

    /// Create a new `Bit` from a boolean value.
    #[inline]
    #[must_use]
    pub const fn new(value: bool) -> Self {
        Bit(value)
    }

    /// Returns `true` if this bit is one.
    #[inline]
    #[must_use]
    pub const fn is_one(self) -> bool {
        self.0
    }

    /// Returns `true` if this bit is zero.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        !self.0
    }

    /// Convert to `u8` (0 or 1).
    #[inline]
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0 as u8
    }

    /// Convert to `usize` (0 or 1).
    #[inline]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Convert to `bool`.
    ///
    /// This is useful when you need a `bool` for macros like `assert!`
    /// that don't use the `Deref` trait.
    #[inline]
    #[must_use]
    pub const fn as_bool(self) -> bool {
        self.0
    }
}

// ============================================================================
// Display and Debug - format as 0/1
// ============================================================================

impl std::fmt::Display for Bit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", u8::from(self.0))
    }
}

impl std::fmt::Debug for Bit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", u8::from(self.0))
    }
}

// ============================================================================
// Deref to bool - allows using Bit in conditions
// ============================================================================

impl Deref for Bit {
    type Target = bool;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// ============================================================================
// From/Into conversions
// ============================================================================

impl From<bool> for Bit {
    #[inline]
    fn from(value: bool) -> Self {
        Bit(value)
    }
}

impl From<Bit> for bool {
    #[inline]
    fn from(bit: Bit) -> Self {
        bit.0
    }
}

impl From<u8> for Bit {
    #[inline]
    fn from(value: u8) -> Self {
        Bit(value != 0)
    }
}

impl From<Bit> for u8 {
    #[inline]
    fn from(bit: Bit) -> Self {
        u8::from(bit.0)
    }
}

impl From<i32> for Bit {
    #[inline]
    fn from(value: i32) -> Self {
        Bit(value != 0)
    }
}

impl From<Bit> for i32 {
    #[inline]
    fn from(bit: Bit) -> Self {
        i32::from(bit.0)
    }
}

impl From<usize> for Bit {
    #[inline]
    fn from(value: usize) -> Self {
        Bit(value != 0)
    }
}

impl From<Bit> for usize {
    #[inline]
    fn from(bit: Bit) -> Self {
        usize::from(bit.0)
    }
}

// ============================================================================
// Bitwise NOT
// ============================================================================

impl Not for Bit {
    type Output = Bit;

    #[inline]
    fn not(self) -> Self::Output {
        Bit(!self.0)
    }
}

impl Not for &Bit {
    type Output = Bit;

    #[inline]
    fn not(self) -> Self::Output {
        Bit(!self.0)
    }
}

// ============================================================================
// Bitwise XOR
// ============================================================================

impl BitXor for Bit {
    type Output = Bit;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Bit(self.0 ^ rhs.0)
    }
}

impl BitXor<&Bit> for Bit {
    type Output = Bit;

    #[inline]
    fn bitxor(self, rhs: &Bit) -> Self::Output {
        Bit(self.0 ^ rhs.0)
    }
}

impl BitXor<Bit> for &Bit {
    type Output = Bit;

    #[inline]
    fn bitxor(self, rhs: Bit) -> Self::Output {
        Bit(self.0 ^ rhs.0)
    }
}

impl BitXor<&Bit> for &Bit {
    type Output = Bit;

    #[inline]
    fn bitxor(self, rhs: &Bit) -> Self::Output {
        Bit(self.0 ^ rhs.0)
    }
}

impl BitXor<bool> for Bit {
    type Output = Bit;

    #[inline]
    fn bitxor(self, rhs: bool) -> Self::Output {
        Bit(self.0 ^ rhs)
    }
}

impl BitXor<Bit> for bool {
    type Output = Bit;

    #[inline]
    fn bitxor(self, rhs: Bit) -> Self::Output {
        Bit(self ^ rhs.0)
    }
}

impl BitXorAssign for Bit {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl BitXorAssign<&Bit> for Bit {
    #[inline]
    fn bitxor_assign(&mut self, rhs: &Bit) {
        self.0 ^= rhs.0;
    }
}

impl BitXorAssign<bool> for Bit {
    #[inline]
    fn bitxor_assign(&mut self, rhs: bool) {
        self.0 ^= rhs;
    }
}

impl BitXorAssign<Bit> for bool {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Bit) {
        *self ^= rhs.0;
    }
}

// ============================================================================
// Bitwise AND
// ============================================================================

impl BitAnd for Bit {
    type Output = Bit;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        Bit(self.0 & rhs.0)
    }
}

impl BitAnd<&Bit> for Bit {
    type Output = Bit;

    #[inline]
    fn bitand(self, rhs: &Bit) -> Self::Output {
        Bit(self.0 & rhs.0)
    }
}

impl BitAnd<Bit> for &Bit {
    type Output = Bit;

    #[inline]
    fn bitand(self, rhs: Bit) -> Self::Output {
        Bit(self.0 & rhs.0)
    }
}

impl BitAnd<&Bit> for &Bit {
    type Output = Bit;

    #[inline]
    fn bitand(self, rhs: &Bit) -> Self::Output {
        Bit(self.0 & rhs.0)
    }
}

impl BitAnd<bool> for Bit {
    type Output = Bit;

    #[inline]
    fn bitand(self, rhs: bool) -> Self::Output {
        Bit(self.0 & rhs)
    }
}

impl BitAnd<Bit> for bool {
    type Output = Bit;

    #[inline]
    fn bitand(self, rhs: Bit) -> Self::Output {
        Bit(self & rhs.0)
    }
}

impl BitAndAssign for Bit {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitAndAssign<&Bit> for Bit {
    #[inline]
    fn bitand_assign(&mut self, rhs: &Bit) {
        self.0 &= rhs.0;
    }
}

impl BitAndAssign<bool> for Bit {
    #[inline]
    fn bitand_assign(&mut self, rhs: bool) {
        self.0 &= rhs;
    }
}

// ============================================================================
// Bitwise OR
// ============================================================================

impl BitOr for Bit {
    type Output = Bit;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Bit(self.0 | rhs.0)
    }
}

impl BitOr<&Bit> for Bit {
    type Output = Bit;

    #[inline]
    fn bitor(self, rhs: &Bit) -> Self::Output {
        Bit(self.0 | rhs.0)
    }
}

impl BitOr<Bit> for &Bit {
    type Output = Bit;

    #[inline]
    fn bitor(self, rhs: Bit) -> Self::Output {
        Bit(self.0 | rhs.0)
    }
}

impl BitOr<&Bit> for &Bit {
    type Output = Bit;

    #[inline]
    fn bitor(self, rhs: &Bit) -> Self::Output {
        Bit(self.0 | rhs.0)
    }
}

impl BitOr<bool> for Bit {
    type Output = Bit;

    #[inline]
    fn bitor(self, rhs: bool) -> Self::Output {
        Bit(self.0 | rhs)
    }
}

impl BitOr<Bit> for bool {
    type Output = Bit;

    #[inline]
    fn bitor(self, rhs: Bit) -> Self::Output {
        Bit(self | rhs.0)
    }
}

impl BitOrAssign for Bit {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitOrAssign<&Bit> for Bit {
    #[inline]
    fn bitor_assign(&mut self, rhs: &Bit) {
        self.0 |= rhs.0;
    }
}

impl BitOrAssign<bool> for Bit {
    #[inline]
    fn bitor_assign(&mut self, rhs: bool) {
        self.0 |= rhs;
    }
}

// ============================================================================
// Comparison with bool
// ============================================================================

impl PartialEq<bool> for Bit {
    #[inline]
    fn eq(&self, other: &bool) -> bool {
        self.0 == *other
    }
}

impl PartialEq<Bit> for bool {
    #[inline]
    fn eq(&self, other: &Bit) -> bool {
        *self == other.0
    }
}

// ============================================================================
// Bits - a collection of Bit values
// ============================================================================

/// A collection of `Bit` values with convenient display and operations.
///
/// This is a newtype wrapper around `Vec<Bit>` that provides:
/// - Display as binary string with LSB on right (standard binary notation)
/// - Convenient methods like `parity()`, `count_ones()`, `len()`
/// - Index access to individual `Bit`s
///
/// # Display Format
///
/// The display format follows standard binary notation where index 0 (LSB)
/// appears on the right. Use `format_lsb_left()` for array order (index 0 on left).
///
/// # Example
///
/// ```
/// use pecos_core::{Bit, Bits};
///
/// // bits[0]=1, bits[1]=1, bits[2]=0
/// let bits = Bits::new(vec![Bit::ONE, Bit::ONE, Bit::ZERO]);
///
/// // Display shows LSB on right: "011" (reading: bits[2], bits[1], bits[0])
/// assert_eq!(format!("{}", bits), "011");
///
/// // format_lsb_left shows array order: "110" (reading: bits[0], bits[1], bits[2])
/// assert_eq!(bits.format_lsb_left(), "110");
///
/// assert_eq!(bits.len(), 3);
/// assert_eq!(bits.count_ones(), 2);
/// assert_eq!(bits.parity(), Bit::ZERO);  // 1 ^ 1 ^ 0 = 0
/// assert_eq!(bits[0], Bit::ONE);
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Bits(pub Vec<Bit>);

impl Bits {
    /// Create a new `Bits` from a vector of bits.
    #[inline]
    #[must_use]
    pub fn new(bits: Vec<Bit>) -> Self {
        Bits(bits)
    }

    /// Create an empty `Bits`.
    #[inline]
    #[must_use]
    pub fn empty() -> Self {
        Bits(Vec::new())
    }

    /// Create a `Bits` with the given capacity.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Bits(Vec::with_capacity(capacity))
    }

    /// Returns the number of bits.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if there are no bits.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Count the number of one bits.
    #[inline]
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.0.iter().filter(|b| b.is_one()).count()
    }

    /// Count the number of zero bits.
    #[inline]
    #[must_use]
    pub fn count_zeros(&self) -> usize {
        self.0.iter().filter(|b| b.is_zero()).count()
    }

    /// Compute the XOR parity of all bits.
    #[inline]
    #[must_use]
    pub fn parity(&self) -> Bit {
        self.0.iter().fold(Bit::ZERO, |acc, &b| acc ^ b)
    }

    /// Returns an iterator over the bits.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Bit> {
        self.0.iter()
    }

    /// Push a bit onto the end.
    #[inline]
    pub fn push(&mut self, bit: Bit) {
        self.0.push(bit);
    }

    /// Get the underlying vector.
    #[inline]
    #[must_use]
    pub fn into_vec(self) -> Vec<Bit> {
        self.0
    }

    /// Get a reference to the underlying slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Bit] {
        &self.0
    }
}

// Display as binary string with LSB on right (standard binary format)
// bits[0] appears on the right, bits[n-1] on the left
// e.g., Bits([ONE, ZERO, ONE]) displays as "101" where bits[0]=1 is rightmost
impl std::fmt::Display for Bits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for bit in self.0.iter().rev() {
            write!(f, "{bit}")?;
        }
        Ok(())
    }
}

// Debug also as binary string for consistency
impl std::fmt::Debug for Bits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\"{self}\"")
    }
}

impl Bits {
    /// Format as a string with index 0 on the left (array order).
    ///
    /// This is the opposite of the default Display which puts index 0
    /// on the right (standard binary notation).
    #[must_use]
    pub fn format_lsb_left(&self) -> String {
        self.0
            .iter()
            .map(|b| if b.is_one() { '1' } else { '0' })
            .collect()
    }
}

// Index access
impl std::ops::Index<usize> for Bits {
    type Output = Bit;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

// IndexMut access
impl std::ops::IndexMut<usize> for Bits {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

// Deref to slice for convenience
impl std::ops::Deref for Bits {
    type Target = [Bit];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// From conversions
impl From<Vec<Bit>> for Bits {
    #[inline]
    fn from(bits: Vec<Bit>) -> Self {
        Bits(bits)
    }
}

impl From<Bits> for Vec<Bit> {
    #[inline]
    fn from(bits: Bits) -> Self {
        bits.0
    }
}

impl From<Vec<bool>> for Bits {
    #[inline]
    fn from(bools: Vec<bool>) -> Self {
        Bits(bools.into_iter().map(Bit::from).collect())
    }
}

impl FromIterator<Bit> for Bits {
    fn from_iter<I: IntoIterator<Item = Bit>>(iter: I) -> Self {
        Bits(iter.into_iter().collect())
    }
}

impl<'a> IntoIterator for &'a Bits {
    type Item = &'a Bit;
    type IntoIter = std::slice::Iter<'a, Bit>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl IntoIterator for Bits {
    type Item = Bit;
    type IntoIter = std::vec::IntoIter<Bit>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Bit::ZERO), "0");
        assert_eq!(format!("{}", Bit::ONE), "1");
        assert_eq!(format!("{:?}", Bit::ZERO), "0");
        assert_eq!(format!("{:?}", Bit::ONE), "1");
    }

    #[test]
    fn test_from_bool() {
        assert_eq!(Bit::from(false), Bit::ZERO);
        assert_eq!(Bit::from(true), Bit::ONE);
    }

    #[test]
    fn test_from_integers() {
        assert_eq!(Bit::from(0u8), Bit::ZERO);
        assert_eq!(Bit::from(1u8), Bit::ONE);
        assert_eq!(Bit::from(42u8), Bit::ONE);

        assert_eq!(Bit::from(0i32), Bit::ZERO);
        assert_eq!(Bit::from(1i32), Bit::ONE);
        assert_eq!(Bit::from(-1i32), Bit::ONE);
    }

    #[test]
    fn test_to_integers() {
        assert_eq!(u8::from(Bit::ZERO), 0);
        assert_eq!(u8::from(Bit::ONE), 1);
        assert_eq!(usize::from(Bit::ONE), 1);
    }

    #[test]
    fn test_not() {
        assert_eq!(!Bit::ZERO, Bit::ONE);
        assert_eq!(!Bit::ONE, Bit::ZERO);
    }

    #[test]
    fn test_xor() {
        assert_eq!(Bit::ZERO ^ Bit::ZERO, Bit::ZERO);
        assert_eq!(Bit::ZERO ^ Bit::ONE, Bit::ONE);
        assert_eq!(Bit::ONE ^ Bit::ZERO, Bit::ONE);
        assert_eq!(Bit::ONE ^ Bit::ONE, Bit::ZERO);

        // XOR with bool
        assert_eq!(Bit::ONE ^ true, Bit::ZERO);
        assert_eq!(true ^ Bit::ONE, Bit::ZERO);
    }

    #[test]
    fn test_and() {
        assert_eq!(Bit::ZERO & Bit::ZERO, Bit::ZERO);
        assert_eq!(Bit::ZERO & Bit::ONE, Bit::ZERO);
        assert_eq!(Bit::ONE & Bit::ZERO, Bit::ZERO);
        assert_eq!(Bit::ONE & Bit::ONE, Bit::ONE);
    }

    #[test]
    fn test_or() {
        assert_eq!(Bit::ZERO | Bit::ZERO, Bit::ZERO);
        assert_eq!(Bit::ZERO | Bit::ONE, Bit::ONE);
        assert_eq!(Bit::ONE | Bit::ZERO, Bit::ONE);
        assert_eq!(Bit::ONE | Bit::ONE, Bit::ONE);
    }

    #[test]
    fn test_assign_ops() {
        let mut b = Bit::ZERO;
        b ^= Bit::ONE;
        assert_eq!(b, Bit::ONE);

        b &= Bit::ONE;
        assert_eq!(b, Bit::ONE);

        b |= Bit::ZERO;
        assert_eq!(b, Bit::ONE);
    }

    #[test]
    fn test_deref() {
        let b = Bit::ONE;
        // Can use in if condition
        if *b {
            // OK
        } else {
            panic!("Deref should work");
        }
    }

    #[test]
    fn test_comparison_with_bool() {
        assert!(Bit::ONE == true);
        assert!(Bit::ZERO == false);
        assert!(true == Bit::ONE);
        assert!(false == Bit::ZERO);
    }

    #[test]
    fn test_vec_debug() {
        let bits = vec![Bit::ONE, Bit::ZERO, Bit::ONE, Bit::ONE, Bit::ZERO];
        // Debug format shows [1, 0, 1, 1, 0] instead of [true, false, ...]
        assert_eq!(format!("{bits:?}"), "[1, 0, 1, 1, 0]");
    }

    #[test]
    fn test_constants() {
        assert!(Bit::ZERO.is_zero());
        assert!(!Bit::ZERO.is_one());
        assert!(Bit::ONE.is_one());
        assert!(!Bit::ONE.is_zero());
    }

    #[test]
    fn test_as_methods() {
        assert_eq!(Bit::ZERO.as_u8(), 0);
        assert_eq!(Bit::ONE.as_u8(), 1);
        assert_eq!(Bit::ZERO.as_usize(), 0);
        assert_eq!(Bit::ONE.as_usize(), 1);
    }

    // ========================================================================
    // Bits tests
    // ========================================================================

    #[test]
    fn test_bits_display() {
        // Display shows LSB (index 0) on the right, like standard binary
        // bits[0]=1, bits[1]=0, bits[2]=1, bits[3]=1, bits[4]=0
        // displays as "01101" (reading left-to-right: bits[4], bits[3], bits[2], bits[1], bits[0])
        let bits = Bits::new(vec![Bit::ONE, Bit::ZERO, Bit::ONE, Bit::ONE, Bit::ZERO]);
        assert_eq!(format!("{bits}"), "01101");
    }

    #[test]
    fn test_bits_debug() {
        // bits[0]=1, bits[1]=0, bits[2]=1 -> "101" (bits[2], bits[1], bits[0])
        let bits = Bits::new(vec![Bit::ONE, Bit::ZERO, Bit::ONE]);
        assert_eq!(format!("{bits:?}"), "\"101\"");
    }

    #[test]
    fn test_bits_format_lsb_left() {
        // format_lsb_left shows index 0 on the left (array order)
        let bits = Bits::new(vec![Bit::ONE, Bit::ZERO, Bit::ONE]);
        assert_eq!(bits.format_lsb_left(), "101");
        // Compare with Display which shows index 0 on the right
        assert_eq!(format!("{bits}"), "101"); // Same in this case (palindrome)

        // Non-palindrome case
        let bits2 = Bits::new(vec![Bit::ONE, Bit::ONE, Bit::ZERO]);
        assert_eq!(bits2.format_lsb_left(), "110"); // index order: [0]=1, [1]=1, [2]=0
        assert_eq!(format!("{bits2}"), "011"); // binary order: [2]=0, [1]=1, [0]=1
    }

    #[test]
    fn test_bits_len() {
        let bits = Bits::new(vec![Bit::ONE, Bit::ZERO, Bit::ONE]);
        assert_eq!(bits.len(), 3);
        assert!(!bits.is_empty());

        let empty = Bits::empty();
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_bits_count() {
        let bits = Bits::new(vec![Bit::ONE, Bit::ZERO, Bit::ONE, Bit::ONE, Bit::ZERO]);
        assert_eq!(bits.count_ones(), 3);
        assert_eq!(bits.count_zeros(), 2);
    }

    #[test]
    fn test_bits_parity() {
        // 1 ^ 0 ^ 1 = 0
        let bits1 = Bits::new(vec![Bit::ONE, Bit::ZERO, Bit::ONE]);
        assert_eq!(bits1.parity(), Bit::ZERO);

        // 1 ^ 1 ^ 1 = 1
        let bits2 = Bits::new(vec![Bit::ONE, Bit::ONE, Bit::ONE]);
        assert_eq!(bits2.parity(), Bit::ONE);

        // empty = 0
        let empty = Bits::empty();
        assert_eq!(empty.parity(), Bit::ZERO);
    }

    #[test]
    fn test_bits_index() {
        let bits = Bits::new(vec![Bit::ONE, Bit::ZERO, Bit::ONE]);
        assert_eq!(bits[0], Bit::ONE);
        assert_eq!(bits[1], Bit::ZERO);
        assert_eq!(bits[2], Bit::ONE);
    }

    #[test]
    fn test_bits_from_vec_bit() {
        // bits[0]=1, bits[1]=0 -> displays as "01" (LSB on right)
        let vec = vec![Bit::ONE, Bit::ZERO];
        let bits: Bits = vec.into();
        assert_eq!(format!("{bits}"), "01");
    }

    #[test]
    fn test_bits_from_vec_bool() {
        // bits[0]=true, bits[1]=false, bits[2]=true -> "101" (palindrome)
        let bools = vec![true, false, true];
        let bits: Bits = bools.into();
        assert_eq!(format!("{bits}"), "101");
    }

    #[test]
    fn test_bits_iter() {
        let bits = Bits::new(vec![Bit::ONE, Bit::ZERO, Bit::ONE]);
        let collected: Vec<&Bit> = bits.iter().collect();
        assert_eq!(collected, vec![&Bit::ONE, &Bit::ZERO, &Bit::ONE]);
    }

    #[test]
    fn test_bits_into_iter() {
        let bits = Bits::new(vec![Bit::ONE, Bit::ZERO]);
        let collected: Vec<Bit> = bits.into_iter().collect();
        assert_eq!(collected, vec![Bit::ONE, Bit::ZERO]);
    }

    #[test]
    fn test_bits_collect() {
        // bits[0]=1, bits[1]=0, bits[2]=1 -> "101" (palindrome)
        let vec = vec![Bit::ONE, Bit::ZERO, Bit::ONE];
        let bits: Bits = vec.into_iter().collect();
        assert_eq!(format!("{bits}"), "101");
    }
}
