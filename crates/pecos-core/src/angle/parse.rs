//! String parsing implementation for angles.
//!
//! This module provides functionality to parse angles from string representations,
//! with support for both radian-based and π-based expressions. The parser provides:
//!
//! - Exact ratio arithmetic for π-based expressions
//! - Support for decimal values in radians
//! - Automatic normalization to [0, 2π) range
//! - Clear error reporting for various failure modes
//!
//! The implementation prioritizes exact representation where possible, using ratio
//! arithmetic for π-based expressions to avoid floating point imprecision.

use super::Angle;
use num_traits::{
    Bounded, FromPrimitive, PrimInt, ToPrimitive, Unsigned, WrappingAdd, WrappingMul, WrappingNeg,
    WrappingSub, Zero,
};
use std::fmt::Debug;
use std::ops::Rem;
use std::str::FromStr;

/// Errors that can occur when parsing angle strings.
///
/// This error type distinguishes between different failure modes:
/// - Invalid format (e.g., malformed expressions)
/// - Numeric parsing failures (invalid numerators/denominators)
/// - Mathematical errors (division by zero)
/// - Range errors (values too large to represent)
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, PartialEq, Eq)]
pub enum ParseAngleError {
    /// Covers all syntax errors (e.g., malformed input, empty string, multiple π symbols)
    InvalidFormat,
    /// Division by zero in fractions
    DivisionByZero,
    /// Angle value exceeds the representable range
    Overflow { value: String },
}

impl std::fmt::Display for ParseAngleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat => write!(f, "invalid angle format"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::Overflow { value } => write!(
                f,
                "angle {value} is too large to represent; use a finite angle with smaller numerator and denominator"
            ),
        }
    }
}

impl std::error::Error for ParseAngleError {}

impl<T> FromStr for Angle<T>
where
    T: Unsigned
        + Copy
        + ToPrimitive
        + FromPrimitive
        + Zero
        + Bounded
        + WrappingAdd
        + WrappingSub
        + WrappingMul
        + WrappingNeg
        + Rem<Output = T>
        + PrimInt
        + Default
        + Debug
        + TryFrom<u128>,
{
    type Err = ParseAngleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_radians(s)
    }
}

fn is_valid_decimal(s: &str) -> bool {
    // String should not start with a decimal point
    if s.starts_with('.') {
        return false;
    }

    let mut dot_count = 0;
    let mut saw_digit = false;
    let chars = s.chars().enumerate();

    for (i, c) in chars {
        match c {
            '-' if i == 0 => {}
            '.' => {
                dot_count += 1;
                if dot_count > 1 {
                    return false;
                }
            }
            c if c.is_ascii_digit() => saw_digit = true,
            _ => return false,
        }
    }
    saw_digit && dot_count <= 1
}

impl<T> Angle<T>
where
    T: Unsigned
        + Copy
        + ToPrimitive
        + FromPrimitive
        + Zero
        + Bounded
        + WrappingAdd
        + WrappingSub
        + WrappingMul
        + WrappingNeg
        + Rem<Output = T>
        + PrimInt
        + Default
        + Debug
        + TryFrom<u128>,
{
    /// Creates an angle from a string representation of radians.
    ///
    /// # Format
    /// Supports several formats for angle specification:
    /// - "π" or "pi" or "PI": Represents π radians (half turn)
    /// - "π/2" or "pi/2": Fractions of π
    /// - "3π" or "3pi": Multiples of π
    /// - "3π/2" or "3pi/2": Complex fractions of π
    /// - "3 π / 2" or "3*π/2": Spaces and * are allowed and ignored
    /// - "1.5" or "-1.5": Raw radian values
    ///
    /// # Normalization
    /// All angles are normalized to the range [0, 2π):
    /// - Negative angles wrap around (e.g., "-π/2" becomes "3π/2")
    /// - Values greater than 2π wrap around modulo 2π
    ///
    /// # Precision
    /// - For pi-based fractions (like "π/2"), uses exact ratio arithmetic
    /// - For decimal values, uses floating point conversion
    ///
    /// # Panics
    /// This function will panic if:
    /// - A decimal point is found but `s.find('.')` returns None
    /// - Pi is detected (via `has_pi`) but neither "pi" nor "π" can be found in the string
    ///
    /// These conditions should be impossible given the logic flow, but are documented
    /// for completeness.
    ///
    /// # Examples
    /// ```
    /// use pecos_core::Angle64;
    /// let three_halves_pi = Angle64::from_str_radians("3π/2").unwrap();
    /// assert_eq!(three_halves_pi, Angle64::THREE_QUARTERS_TURN);
    ///
    /// // Negative angles wrap around
    /// let neg_half_pi = Angle64::from_str_radians("-π/2").unwrap();
    /// assert_eq!(neg_half_pi, Angle64::THREE_QUARTERS_TURN);
    /// ```
    ///
    /// # Errors
    /// Returns `ParseAngleError` if:
    /// - The string format is invalid (e.g., malformed expressions)
    /// - The numerator or denominator can't be parsed as numbers
    /// - The denominator is zero
    /// - The resulting angle would overflow the target type
    #[allow(clippy::cast_precision_loss)]
    pub fn from_str_radians(s: &str) -> Result<Self, ParseAngleError> {
        let s = s.trim().replace([' ', '*'], "").to_lowercase();

        if s.is_empty() {
            return Err(ParseAngleError::InvalidFormat);
        }

        // Check if the input starts with "/", ends with "/", or starts with "-/"
        if s.starts_with('/') || s.ends_with('/') || s.starts_with("-/") {
            return Err(ParseAngleError::InvalidFormat);
        }

        // Check for multiple π symbols (existing logic)
        if s.matches('π').count() > 1 || s.matches("pi").count() > 1 {
            return Err(ParseAngleError::InvalidFormat);
        }

        // First check if it's just "pi" or "π" or "-pi" or "-π"
        if s == "pi" || s == "π" {
            return Ok(Self::from_turn_ratio(1, 2));
        } else if s == "-pi" || s == "-π" {
            return Ok(Self::from_turn_ratio(-1, 2));
        }

        // Handle decimal numbers (with or without pi)
        if s.contains('.') {
            return Self::try_from_radians(Self::parse_decimal(&s)?).map_err(|error| {
                ParseAngleError::Overflow {
                    value: format!("{} {} (input {s:?})", error.value, error.unit),
                }
            });
        }

        // Split into numerator and denominator parts
        let (num_part, den_part) = if let Some((n, d)) = s.split_once('/') {
            (n, Some(d))
        } else {
            (s.as_str(), None)
        };

        // Parse numerator, handling pi/π multiplier
        let (num_val, has_pi) = if num_part.contains("pi") || num_part.contains('π') {
            let n = num_part.replace("pi", "").replace('π', "");
            (Self::parse_pi_numerator(&n)?, true)
        } else if let Ok(val) = num_part.parse::<i64>() {
            (val, false)
        } else {
            let is_valid = num_part.starts_with('-')
                && num_part[1..].chars().all(|c| c.is_ascii_digit())
                || num_part.chars().all(|c| c.is_ascii_digit());
            if is_valid {
                return Err(ParseAngleError::Overflow { value: s.clone() });
            }
            return Err(ParseAngleError::InvalidFormat);
        };

        // Parse denominator
        let den_val = if let Some(d) = den_part {
            match d.parse::<i64>() {
                Ok(den) => {
                    if den == 0 {
                        return Err(ParseAngleError::DivisionByZero);
                    }
                    den
                }
                Err(_) => return Err(ParseAngleError::InvalidFormat),
            }
        } else {
            1
        };

        // Check for potential overflow in multiplication when has_pi is true
        if has_pi && (num_val.checked_mul(2).is_none() || den_val.checked_mul(2).is_none()) {
            return Err(ParseAngleError::Overflow { value: s.clone() });
        }

        // Convert to angle using appropriate method
        if has_pi {
            Ok(Self::from_turn_ratio(num_val, den_val * 2))
        } else {
            let radians = num_val as f64 / den_val as f64;
            Self::try_from_radians(radians).map_err(|error| ParseAngleError::Overflow {
                value: error.to_string(),
            })
        }
    }

    /// Handle cases involving decimal points
    fn parse_decimal(s: &str) -> Result<f64, ParseAngleError> {
        if let Some((_, _)) = s.split_once('/') {
            return Err(ParseAngleError::InvalidFormat);
        }

        let has_pi = s.contains("pi") || s.contains('π');

        // If pi/π comes before the decimal point, it's invalid
        if has_pi && s.contains('.') {
            // Both pi_pos and dot_pos are guaranteed to exist because of the has_pi and contains('.') checks
            let pi_pos = s
                .find("pi")
                .or_else(|| s.find('π'))
                .expect("pi position not found despite has_pi being true");
            let dot_pos = s
                .find('.')
                .expect("decimal point position not found despite contains('.') being true");
            if pi_pos < dot_pos {
                return Err(ParseAngleError::InvalidFormat);
            }
        }

        if has_pi {
            // Remove pi/π and parse number first
            let n = s.replace("pi", "").replace('π', "").trim().to_string();
            if !is_valid_decimal(&n) {
                return Err(ParseAngleError::InvalidFormat);
            }
            let value = n
                .parse::<f64>()
                .map_err(|_| ParseAngleError::InvalidFormat)?;
            if !value.is_finite() {
                return Err(ParseAngleError::Overflow {
                    value: s.to_string(),
                });
            }
            Ok(value * std::f64::consts::PI)
        } else {
            if !is_valid_decimal(s) {
                return Err(ParseAngleError::InvalidFormat);
            }
            let value = s
                .parse::<f64>()
                .map_err(|_| ParseAngleError::InvalidFormat)?;
            if !value.is_finite() {
                return Err(ParseAngleError::Overflow {
                    value: s.to_string(),
                });
            }
            Ok(value)
        }
    }

    /// Handle parsing of numerator part when pi is present
    fn parse_pi_numerator(n: &str) -> Result<i64, ParseAngleError> {
        if n.is_empty() {
            return Ok(1);
        }
        if n == "-" {
            return Ok(-1);
        }
        // Try parsing - if it fails, determine if it's invalid format or overflow
        if let Ok(val) = n.parse::<i64>() {
            Ok(val)
        } else {
            // Check if it's a valid number format that's just too big
            let is_valid = n.starts_with('-') && n[1..].chars().all(|c| c.is_ascii_digit())
                || n.chars().all(|c| c.is_ascii_digit());
            if is_valid {
                Err(ParseAngleError::Overflow {
                    value: n.to_string(),
                })
            } else {
                Err(ParseAngleError::InvalidFormat)
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Angle64;
    use std::f64::consts::PI;

    #[test]
    fn test_parse_basic_pi() {
        assert_eq!(Angle64::from_str_radians("pi").unwrap(), Angle64::HALF_TURN);
        assert_eq!(Angle64::from_str_radians("π").unwrap(), Angle64::HALF_TURN);
        assert_eq!(Angle64::from_str_radians("PI").unwrap(), Angle64::HALF_TURN);
    }

    #[test]
    fn test_parse_fractions() {
        assert_eq!(
            Angle64::from_str_radians("pi/2").unwrap(),
            Angle64::QUARTER_TURN
        );
        assert_eq!(
            Angle64::from_str_radians("π/2").unwrap(),
            Angle64::QUARTER_TURN
        );
        assert_eq!(
            Angle64::from_str_radians("PI/2").unwrap(),
            Angle64::QUARTER_TURN
        );
    }

    #[test]
    fn test_parse_multiples() {
        assert_eq!(
            Angle64::from_str_radians("2pi").unwrap(),
            Angle64::FULL_TURN
        );
        assert_eq!(Angle64::from_str_radians("2π").unwrap(), Angle64::FULL_TURN);
        assert_eq!(
            Angle64::from_str_radians("2PI").unwrap(),
            Angle64::FULL_TURN
        );
    }

    #[test]
    fn test_parse_complex() {
        assert_eq!(
            Angle64::from_str_radians("3pi/2").unwrap(),
            Angle64::THREE_QUARTERS_TURN
        );
        assert_eq!(
            Angle64::from_str_radians("3π/2").unwrap(),
            Angle64::THREE_QUARTERS_TURN
        );
        assert_eq!(
            Angle64::from_str_radians("3*pi/2").unwrap(),
            Angle64::THREE_QUARTERS_TURN
        );
    }

    #[test]
    fn test_parse_with_spaces() {
        assert_eq!(
            Angle64::from_str_radians("3 pi / 2").unwrap(),
            Angle64::THREE_QUARTERS_TURN
        );
        assert_eq!(
            Angle64::from_str_radians("3 * pi / 2").unwrap(),
            Angle64::THREE_QUARTERS_TURN
        );
        assert_eq!(
            Angle64::from_str_radians("  pi  ").unwrap(),
            Angle64::HALF_TURN
        );
    }

    #[test]
    fn test_parse_non_pi() {
        let angle = Angle64::from_str_radians("1.5").unwrap();
        assert!((angle.to_radians() - 1.5).abs() < 1e-10);
    }
    #[test]
    fn test_parse_pi_decimal() {
        // Test various decimal * pi formats
        let test_cases = [
            ("1.5 * pi", 1.5),
            ("0.5pi", 0.5),
            ("0.25 * pi", 0.25),
            ("0.3 pi", 0.3),
            ("-0.5pi", 1.5),
            ("0.75 * pi", 0.75),
        ];

        for (input, multiplier) in test_cases {
            let angle = Angle64::from_str_radians(input).unwrap();
            let actual_radians = angle.to_radians();
            let expected_radians = multiplier * PI;
            assert!(
                (actual_radians - expected_radians).abs() < 1e-10,
                "Failed for input: {input}"
            );
        }
    }

    // Make sure we also handle π symbol the same way
    #[test]
    fn test_parse_pi_symbol_decimal() {
        // Test various decimal * π formats
        let test_cases = [
            ("1.5 * π", 1.5),
            ("0.5π", 0.5),
            ("0.25 * π", 0.25),
            ("0.3 π", 0.3),
            ("-0.5π", 1.5),
            ("0.75 * π", 0.75),
        ];

        for (input, multiplier) in test_cases {
            let angle = Angle64::from_str_radians(input).unwrap();
            assert!(
                (angle.to_radians() - multiplier * PI).abs() < 1e-10,
                "Failed for input: {input}"
            );
        }
    }

    #[test]
    fn test_parse_decimal_errors() {
        // These should all be invalid formats
        assert_eq!(
            Angle64::from_str_radians("1.5.2pi"),
            Err(ParseAngleError::InvalidFormat),
            "multiple decimals should be invalid"
        );
        assert_eq!(
            Angle64::from_str_radians("pi.2"),
            Err(ParseAngleError::InvalidFormat),
            "decimal after pi should be invalid"
        );
        assert_eq!(
            Angle64::from_str_radians("1.2pi/2"),
            Err(ParseAngleError::InvalidFormat),
            "decimal with fraction should be invalid"
        );
    }

    #[test]
    fn test_parse_errors() {
        assert_eq!(
            Angle64::from_str_radians("invalid"),
            Err(ParseAngleError::InvalidFormat)
        );
        assert_eq!(
            Angle64::from_str_radians("pi/0"),
            Err(ParseAngleError::DivisionByZero)
        );
        assert_eq!(
            Angle64::from_str_radians("pi/invalid"),
            Err(ParseAngleError::InvalidFormat)
        );
    }

    #[test]
    fn test_parse_fromstr_trait() {
        let angle: Angle64 = "pi/2".parse().unwrap();
        assert_eq!(angle, Angle64::QUARTER_TURN);
    }

    #[test]
    fn test_parse_edge_cases() {
        // Very large numbers
        assert!(
            Angle64::from_str_radians(&format!("{}pi", i64::MAX)).is_err(),
            "failed to deal with large numbers"
        );

        // Negative numbers - Note: -pi/2 gets normalized to 3pi/2
        let neg_quarter = Angle64::from_str_radians("-pi/2").unwrap();
        assert_eq!(
            neg_quarter,
            Angle64::THREE_QUARTERS_TURN,
            "failed to handle negative numbers"
        );

        // Zero
        assert!(
            Angle64::from_str_radians("0").unwrap().is_zero(),
            "failed to handle zero (0)"
        );
        assert!(
            Angle64::from_str_radians("0pi").unwrap().is_zero(),
            "failed to handle zero (0pi)"
        );

        // Just negative sign
        assert_eq!(
            Angle64::from_str_radians("-pi").unwrap(),
            Angle64::HALF_TURN,
            "failed to handle -pi"
        );
    }

    #[test]
    fn test_invalid_formats_multiple_pi() {
        // Multiple pi symbols
        assert_eq!(
            Angle64::from_str_radians("pipi"),
            Err(ParseAngleError::InvalidFormat),
            "multiple consecutive pi symbols should be invalid"
        );
        assert_eq!(
            Angle64::from_str_radians("2pi3pi"),
            Err(ParseAngleError::InvalidFormat),
            "multiple pi symbols should be invalid"
        );
        assert_eq!(
            Angle64::from_str_radians("πpiπ"),
            Err(ParseAngleError::InvalidFormat),
            "mixed pi symbols should be invalid"
        );
    }

    #[test]
    fn test_invalid_formats_fraction_bars() {
        // Multiple or misplaced fraction bars
        assert_eq!(
            Angle64::from_str_radians("pi/2/3"),
            Err(ParseAngleError::InvalidFormat),
            "multiple fraction bars should be invalid"
        );
        assert_eq!(
            Angle64::from_str_radians("1/2/3"),
            Err(ParseAngleError::InvalidFormat),
            "multiple fraction bars without pi should be invalid"
        );
        assert_eq!(
            Angle64::from_str_radians("/2"),
            Err(ParseAngleError::InvalidFormat),
            "leading fraction bar should be invalid"
        );
        assert_eq!(
            Angle64::from_str_radians("2/"),
            Err(ParseAngleError::InvalidFormat),
            "trailing fraction bar should be invalid"
        );
        assert_eq!(
            Angle64::from_str_radians("2//3"),
            Err(ParseAngleError::InvalidFormat),
            "consecutive fraction bars should be invalid"
        );
    }

    #[test]
    fn test_invalid_formats_empty_string() {
        // Empty strings and whitespace
        assert_eq!(
            Angle64::from_str_radians(""),
            Err(ParseAngleError::InvalidFormat),
            "empty string should be invalid"
        );
        assert_eq!(
            Angle64::from_str_radians("   "),
            Err(ParseAngleError::InvalidFormat),
            "whitespace only should be invalid"
        );
    }

    #[test]
    fn test_invalid_formats_decimal_fraction() {
        // Mixed decimal and fraction forms
        assert_eq!(
            Angle64::from_str_radians("1.2/3"),
            Err(ParseAngleError::InvalidFormat),
            "decimal in fraction numerator should be invalid"
        );
        assert_eq!(
            Angle64::from_str_radians("1/2.3"),
            Err(ParseAngleError::InvalidFormat),
            "decimal in fraction denominator should be invalid"
        );
        assert_eq!(
            Angle64::from_str_radians("1.2pi/3"),
            Err(ParseAngleError::InvalidFormat),
            "decimal with pi in fraction should be invalid"
        );
    }

    #[test]
    fn test_invalid_starts_with_slash() {
        assert_eq!(
            Angle64::from_str_radians("/3"),
            Err(ParseAngleError::InvalidFormat),
            "front slash should be invalid"
        );
    }

    #[test]
    fn test_invalid_ends_with_slash() {
        assert_eq!(
            Angle64::from_str_radians("3/"),
            Err(ParseAngleError::InvalidFormat),
            "trailing slash should be invalid"
        );
    }

    #[test]
    fn test_invalid_starts_with_negative_slash() {
        assert_eq!(
            Angle64::from_str_radians("-/3"),
            Err(ParseAngleError::InvalidFormat),
            "front negative slash should be invalid"
        );
    }
}
