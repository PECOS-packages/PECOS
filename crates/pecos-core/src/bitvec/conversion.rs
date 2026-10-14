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

//! Conversion operations for `BitVec` values

use bitvec::prelude::*;

use super::arithmetic::add_extend;
use super::bitwise::shift_left_extend;

/// Convert a u32 to minimal `BitVec` representation
///
/// # Arguments
/// * `value` - The u32 value to convert
///
/// # Returns
/// A `BitVec` with the minimal number of bits needed to represent the value
#[must_use]
pub fn from_u32(value: u32) -> BitVec<u8, Lsb0> {
    if value == 0 {
        let mut result = BitVec::new();
        result.push(false);
        return result;
    }

    let bits_needed = 32 - value.leading_zeros();
    let mut result = BitVec::with_capacity(bits_needed as usize);

    for i in 0..bits_needed {
        result.push((value >> i) & 1 != 0);
    }

    result
}

/// Convert a `BitVec` to a u32 value if possible
///
/// # Arguments
/// * `bitvec` - The `BitVec` to convert
///
/// # Returns
/// `Some(value)` if conversion is possible, `None` if the value is too large or negative
#[must_use]
pub fn to_u32(bitvec: &BitVec<u8, Lsb0>) -> Option<u32> {
    if bitvec.is_empty() {
        return Some(0);
    }

    // Check if it's negative (sign bit set)
    if bitvec.len() > 32 && bitvec.last().as_deref().copied().unwrap_or(false) {
        return None; // Negative value
    }

    let mut result = 0u32;
    for (i, bit) in bitvec.iter().take(32).enumerate() {
        if *bit {
            result |= 1 << i;
        }
    }

    Some(result)
}

/// Convert a `BitVec` to an i32 value (interprets as signed two's complement)
///
/// # Arguments
/// * `bitvec` - The `BitVec` to convert
///
/// # Returns
/// The signed i32 value with proper sign extension
#[must_use]
pub fn to_i32(bitvec: &BitVec<u8, Lsb0>) -> i32 {
    if bitvec.is_empty() {
        return 0;
    }

    // Check sign bit
    let is_negative = bitvec.last().as_deref().copied().unwrap_or(false);

    if bitvec.len() <= 32 {
        // Can fit in i32
        let mut value = 0i32;
        for (i, bit) in bitvec.iter().enumerate() {
            if i < 32 && *bit {
                value |= 1 << i;
            }
        }

        // If negative and less than 32 bits, sign extend
        if is_negative && bitvec.len() < 32 {
            // Set all bits from bitvec.len() to 31
            for i in bitvec.len()..32 {
                value |= 1 << i;
            }
        }

        value
    } else {
        // Truncate to 32 bits, preserving sign
        let mut value = 0i32;
        for i in 0..31 {
            if bitvec[i] {
                value |= 1 << i;
            }
        }
        // Set sign bit
        if is_negative {
            value |= 1 << 31;
        }
        value
    }
}

/// Read a `BitVec` as an unsigned little-endian `u64`.
///
/// Returns `None` if any bit at index 64 or above is set. Leading zero bits
/// are permitted, and an empty vector represents zero.
#[must_use]
pub fn to_u64(bitvec: &BitVec<u8, Lsb0>) -> Option<u64> {
    let mut value = 0u64;
    for index in bitvec.iter_ones() {
        value |= 1u64.checked_shl(u32::try_from(index).ok()?)?;
    }
    Some(value)
}

/// Convert a `BitVec` to an i64 value (interprets as signed two's complement)
///
/// # Arguments
/// * `bitvec` - The `BitVec` to convert
///
/// # Returns
/// The signed i64 value with proper sign extension
#[must_use]
pub fn to_i64(bitvec: &BitVec<u8, Lsb0>) -> i64 {
    if bitvec.is_empty() {
        return 0;
    }

    // Check sign bit
    let is_negative = bitvec.last().as_deref().copied().unwrap_or(false);

    if bitvec.len() <= 64 {
        // Can fit in i64
        let mut value = 0i64;
        for (i, bit) in bitvec.iter().enumerate() {
            if i < 64 && *bit {
                value |= 1 << i;
            }
        }

        // If negative and less than 64 bits, sign extend
        if is_negative && bitvec.len() < 64 {
            // Set all bits from bitvec.len() to 63
            for i in bitvec.len()..64 {
                value |= 1 << i;
            }
        }

        value
    } else {
        // Truncate to 64 bits, preserving sign
        let mut value = 0i64;
        for i in 0..63 {
            if bitvec[i] {
                value |= 1 << i;
            }
        }
        // Set sign bit
        if is_negative {
            value |= 1 << 63;
        }
        value
    }
}

/// Convert a `BitVec` to an i128 value (interprets as signed two's complement)
///
/// # Arguments
/// * `bitvec` - The `BitVec` to convert
///
/// # Returns
/// The signed i128 value with proper sign extension
#[must_use]
pub fn to_i128(bitvec: &BitVec<u8, Lsb0>) -> i128 {
    if bitvec.is_empty() {
        return 0;
    }

    // Check sign bit
    let is_negative = bitvec.last().as_deref().copied().unwrap_or(false);

    if bitvec.len() <= 128 {
        // Can fit in i128
        let mut value = 0i128;
        for (i, bit) in bitvec.iter().enumerate() {
            if i < 128 && *bit {
                value |= 1 << i;
            }
        }

        // If negative and less than 128 bits, sign extend
        if is_negative && bitvec.len() < 128 {
            // Set all bits from bitvec.len() to 127
            for i in bitvec.len()..128 {
                value |= 1 << i;
            }
        }

        value
    } else {
        // Truncate to 128 bits, preserving sign
        let mut value = 0i128;
        for i in 0..127 {
            if bitvec[i] {
                value |= 1 << i;
            }
        }
        // Set sign bit
        if is_negative {
            value |= 1 << 127;
        }
        value
    }
}

/// Parse an arbitrary-length decimal integer string into a `BitVec`
/// This only handles positive integers - negative signs should be handled as unary operations
///
/// # Arguments
/// * `s` - The decimal string to parse
///
/// # Returns
/// `Ok(BitVec)` if parsing succeeds, `Err` if the string contains invalid characters
///
/// # Errors
/// Returns an error if the string contains invalid decimal digits or is empty
pub fn parse_decimal_string(s: &str) -> Result<BitVec<u8, Lsb0>, String> {
    let s = s.trim();

    // We should only receive positive integers here
    // Negative numbers should be handled as unary operations
    if s.starts_with('-') {
        return Err(format!(
            "parse_decimal_string should only receive positive integers, got: {s}"
        ));
    }

    // Handle empty string
    if s.is_empty() {
        return Err("Empty string".to_string());
    }

    // Start with zero
    let mut result = BitVec::new();
    result.push(false);

    for ch in s.chars() {
        if let Some(digit) = ch.to_digit(10) {
            // Multiply result by 10 (= * 8 + * 2)
            let times_8 = shift_left_extend(&result, 3); // * 8
            let times_2 = shift_left_extend(&result, 1); // * 2

            // result = times_8 + times_2
            result = add_extend(&times_8, &times_2);

            // Add the digit
            if digit > 0 {
                let digit_bits = from_u32(digit);
                result = add_extend(&result, &digit_bits);
            }
        } else {
            return Err(format!("Invalid character '{ch}' in number"));
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitvec::display::to_decimal_string;

    #[test]
    fn test_to_u64_unsigned_top_bit() {
        assert_eq!(to_u64(&bitvec![u8, Lsb0; 0, 0, 0, 1]), Some(8));
        assert_eq!(to_u64(&bitvec![u8, Lsb0; 1]), Some(1));
        assert_eq!(to_u64(&BitVec::repeat(true, 64)), Some(u64::MAX));
    }

    #[test]
    fn test_to_u64_empty() {
        assert_eq!(to_u64(&BitVec::new()), Some(0));
    }

    #[test]
    fn test_to_u64_leading_zeros() {
        let mut value = BitVec::repeat(true, 64);
        value.resize(65, false);
        assert_eq!(to_u64(&value), Some(u64::MAX));
        value.resize(129, false);
        assert_eq!(to_u64(&value), Some(u64::MAX));
        assert_eq!(to_u64(&BitVec::repeat(false, 129)), Some(0));
    }

    #[test]
    fn test_to_u64_overflow() {
        for (width, high_bit) in [(65, 64), (129, 64), (129, 128)] {
            let mut value = BitVec::repeat(false, width);
            value.set(high_bit, true);
            assert_eq!(to_u64(&value), None);
        }
    }

    #[test]
    fn test_from_u32() {
        // Test zero
        let result = from_u32(0);
        assert_eq!(result.len(), 1);
        assert!(!result[0]);

        // Test small number
        let result = from_u32(5); // 101 in binary
        assert_eq!(result.len(), 3);
        assert!(result[0]); // LSB
        assert!(!result[1]);
        assert!(result[2]); // MSB
    }

    #[test]
    fn test_parse_decimal_string() {
        // Test simple number
        let result = parse_decimal_string("123").unwrap();
        assert_eq!(to_decimal_string(&result), "123");

        // Test zero
        let result = parse_decimal_string("0").unwrap();
        assert_eq!(to_decimal_string(&result), "0");

        // Test large number
        let result = parse_decimal_string("1000").unwrap();
        assert_eq!(to_decimal_string(&result), "1000");

        // Test error cases
        assert!(parse_decimal_string("").is_err());
        assert!(parse_decimal_string("-123").is_err());
        assert!(parse_decimal_string("12a3").is_err());
    }

    #[test]
    fn test_signed_conversions() {
        // Test positive small number
        let mut bv = BitVec::<u8, Lsb0>::new();
        bv.extend([true, false, true, false]); // 5 in 4-bit binary (LSB first)
        assert_eq!(to_i32(&bv), 5);
        assert_eq!(to_i64(&bv), 5);
        assert_eq!(to_i128(&bv), 5);

        // Test negative number (4-bit two's complement)
        let mut bv = BitVec::<u8, Lsb0>::new();
        bv.extend([true, true, true, true]); // -1 in 4-bit two's complement
        assert_eq!(to_i32(&bv), -1);
        assert_eq!(to_i64(&bv), -1);
        assert_eq!(to_i128(&bv), -1);

        // Test -5 in 4-bit two's complement: 1011
        let mut bv = BitVec::<u8, Lsb0>::new();
        bv.extend([true, true, false, true]); // -5 in 4-bit two's complement
        assert_eq!(to_i32(&bv), -5);
        assert_eq!(to_i64(&bv), -5);
        assert_eq!(to_i128(&bv), -5);

        // Test empty BitVec
        let bv = BitVec::<u8, Lsb0>::new();
        assert_eq!(to_i32(&bv), 0);
        assert_eq!(to_i64(&bv), 0);
        assert_eq!(to_i128(&bv), 0);

        // Test large positive number that fits in i32
        let mut bv = BitVec::<u8, Lsb0>::new();
        // 2147483647 (i32::MAX) in binary
        for _ in 0..31 {
            bv.push(true);
        }
        bv.push(false); // Sign bit
        assert_eq!(to_i32(&bv), i32::MAX);
        assert_eq!(to_i64(&bv), i64::from(i32::MAX));

        // Test truncation of large BitVec to i32
        let mut bv = BitVec::<u8, Lsb0>::new();
        // Create a 64-bit value that will be truncated
        for _ in 0..64 {
            bv.push(true);
        }
        bv.push(false); // Sign bit
        // When truncated to 32 bits, we get all 1s in lower 31 bits with sign bit 0
        assert_eq!(to_i32(&bv), i32::MAX);
    }

    #[test]
    fn test_sign_extension() {
        // Test sign extension for positive number
        let mut bv = BitVec::<u8, Lsb0>::new();
        bv.extend([true, false, true, false]); // 5 in 4-bit (positive)
        assert_eq!(to_i32(&bv), 5);
        assert_eq!(to_i64(&bv), 5);

        // Test sign extension for negative 8-bit number
        let mut bv = BitVec::<u8, Lsb0>::new();
        // -128 in 8-bit: 10000000 (LSB first: 00000001)
        bv.push(false);
        for _ in 1..7 {
            bv.push(false);
        }
        bv.push(true); // Sign bit
        assert_eq!(to_i32(&bv), -128);
        assert_eq!(to_i64(&bv), -128);
        assert_eq!(to_i128(&bv), -128);
    }
}
