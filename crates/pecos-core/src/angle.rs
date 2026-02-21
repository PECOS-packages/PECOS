//! # Angle: Fixed-Point Representation for Rotations
//!
//! The `Angle` struct provides a fixed-point fractional representation for angles in the range `[0, 2^n)`,
//! where `n` is the bit width of the underlying unsigned integer (`u32`, `u64`, or `u128`).
//! It is optimized for fast, modular arithmetic operations, suitable for applications like simulations,
//! graphics, and scientific computing.
//!
//! ## Key Features
//! - Fixed-point representation with efficient modular arithmetic.
//! - Common angle constants: `ZERO`, `QUARTER_TURN`, `HALF_TURN`, `THREE_QUARTERS_TURN`, `FULL_TURN`.
//! - Arithmetic operations: addition, subtraction, multiplication, division.
//! - Conversion to radians.
//!
//! ## Example Usage
//! ```rust
//! use pecos_core::Angle64;
//! let quarter_turn = Angle64::QUARTER_TURN;
//! let half_turn = quarter_turn + quarter_turn;
//! assert_eq!(half_turn.fraction(), Angle64::HALF_TURN.fraction());
//!
//! let radians = half_turn.to_radians();
//! assert!((radians - std::f64::consts::PI).abs() < 1e-6);
//! ```
mod macros;
mod parse;
mod trig;

use num_traits::{
    Bounded, FromPrimitive, PrimInt, ToPrimitive, Unsigned, WrappingAdd, WrappingMul, WrappingNeg,
    WrappingSub, Zero,
};
pub use parse::ParseAngleError;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

/// Alias for `Angle` with an 8-bit unsigned integer.
#[allow(clippy::module_name_repetitions)]
pub type Angle8 = Angle<u8>;

/// Alias for `Angle` with a 16-bit unsigned integer.
#[allow(clippy::module_name_repetitions)]
pub type Angle16 = Angle<u16>;

/// Alias for `Angle` with a 32-bit unsigned integer.
#[allow(clippy::module_name_repetitions)]
pub type Angle32 = Angle<u32>;

/// Alias for `Angle` with a 64-bit unsigned integer.
#[allow(clippy::module_name_repetitions)]
pub type Angle64 = Angle<u64>;

/// Alias for `Angle` with a 128-bit unsigned integer.
#[allow(clippy::module_name_repetitions)]
pub type Angle128 = Angle<u128>;

/// A fixed-point representation for angles, stored as a fraction of a full turn (2π radians).
///
/// # Implementation Details
/// - Uses a fixed-point representation in the range [0, 2^n) for an n-bit unsigned integer
/// - The full range of the integer type represents one complete turn (2π radians)
/// - All operations (addition, subtraction, etc.) automatically wrap around at full turns
/// - Provides both exact ratio-based and floating-point conversion methods
///
/// # Precision
/// The precision depends on the bit width of the underlying type:
/// - u8: 8 bits (256 distinct angles)
/// - u16: 16 bits (65,536 distinct angles)
/// - u32: 32 bits (~4.3 billion distinct angles)
/// - u64: 64 bits (high precision)
/// - u128: 128 bits (extremely high precision)
///
/// # Examples
/// ```rust
/// use pecos_core::Angle64;
///
/// // Basic arithmetic wraps around automatically
/// let quarter = Angle64::QUARTER_TURN;
/// let half = quarter + quarter;
/// assert_eq!(half, Angle64::HALF_TURN);
///
/// // Conversion to radians for trigonometry
/// let radians = half.to_radians();
/// assert!((radians - std::f64::consts::PI).abs() < 1e-6);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, PartialOrd, Ord)]
pub struct Angle<T: Unsigned + Copy> {
    fraction: T, // Fixed-point fractional representation in [0, 2^n) turns
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
        + WrappingNeg
        + Rem<Output = T>,
{
    /// Creates a new angle from a fraction of a full turn.
    /// The fraction is interpreted as a fixed-point number where the full range
    /// of T represents one full turn.
    #[inline]
    pub const fn new(fraction: T) -> Self {
        Self { fraction }
    }

    /// Returns the bit (fixed-point) representation of the angle
    pub fn fraction(&self) -> T {
        self.fraction
    }

    /// Converts the angle to radians in `[0, 2π)`.
    ///
    /// # Panics
    /// This function will panic if the conversion of `fraction` or `max_value` to `f64` fails.
    pub fn to_radians(&self) -> f64 {
        let max_value = T::max_value()
            .to_f64()
            .expect("Failed to convert max_value to f64");
        self.fraction
            .to_f64()
            .expect("Failed to convert fraction to f64")
            / max_value
            * std::f64::consts::TAU
    }

    /// Converts the angle to radians in `(-π, π]`.
    ///
    /// This is the signed (principal-value) form. It is essential for rotation
    /// gates that use half-angle trig (`θ/2`): because `exp(-iθG/2)` has period
    /// 4π while `Angle` wraps at 2π, using the unsigned `[0, 2π)` form for
    /// half-angle computation introduces a global phase of −1 for angles whose
    /// unsigned representation exceeds π.  The signed form avoids this.
    ///
    /// # Panics
    /// This function will panic if the conversion of `fraction` or `max_value` to `f64` fails.
    pub fn to_radians_signed(&self) -> f64 {
        let r = self.to_radians();
        if r > std::f64::consts::PI {
            r - std::f64::consts::TAU
        } else {
            r
        }
    }

    /// Creates an angle from a value in radians.
    ///
    /// # Panics
    /// This function will panic if the conversion from f64 to the target type fails.
    #[inline]
    #[must_use]
    pub fn from_radians(radians: f64) -> Self {
        const TAU: f64 = std::f64::consts::TAU;

        // Normalize the input to [0, 2π)
        let fraction = ((radians.rem_euclid(TAU) / TAU)
            * T::max_value()
                .to_f64()
                .expect("Conversion of max_value to f64 failed"))
        .round();

        Self {
            fraction: T::from_f64(fraction).expect("Conversion of fraction to target type failed"),
        }
    }

    /// Creates an angle from a value in turns.
    ///
    /// # Panics
    /// This function will panic if the conversion from f64 to the target type fails.
    #[inline]
    #[must_use]
    pub fn from_turns(turns: f64) -> Self {
        // Normalize the input to [0, 1) turns
        let normalized_turns = turns.rem_euclid(1.0);

        let fraction = (normalized_turns
            * T::max_value()
                .to_f64()
                .expect("Conversion of max_value to f64 failed"))
        .round();

        Self {
            fraction: T::from_f64(fraction).expect("Conversion of fraction to target type failed"),
        }
    }

    /// Returns true if this angle is exactly 0.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.fraction == T::zero()
    }

    /// Normalizes the angle to be within [0, 2π).
    /// This is a no-op for this implementation since the fixed-point representation is always normalized.
    #[inline]
    #[must_use]
    pub fn normalize(&self) -> Self {
        *self
    }
}

impl<T> Angle<T>
where
    T: TryFrom<u128> + Default + Unsigned + Copy + PrimInt,
{
    /// Creates an `Angle` from a ratio of a turn.
    ///
    /// This method calculates the angle as `numerator / denominator` of a turn,
    /// where a full turn corresponds to the maximum fixed-point value.
    ///
    /// # Panics
    /// Code will panic if:
    /// - The denominator is zero.
    /// - The numerator is too large to represent in the underlying type.
    #[must_use]
    pub fn from_turn_ratio(mut numerator: i64, mut denominator: i64) -> Self {
        // Normalize denominator and handle full-turn case inline.
        if denominator < 0 {
            numerator = -numerator;
            denominator = -denominator;
        } else if numerator.abs() == denominator {
            return Self {
                fraction: T::zero(), // 0 turns
            };
        }

        let abs_numerator = u128::from(numerator.unsigned_abs());
        let abs_denominator = u128::from(denominator.unsigned_abs());

        let scaling_factor = 1_u128 << u64::from(T::default().count_zeros());

        // Perform scaling and rounding using bit-shifting.
        let mut fraction =
            (abs_numerator * scaling_factor + (abs_denominator >> 1)) / abs_denominator;

        // Apply the sign of the numerator.
        if numerator < 0 {
            fraction = scaling_factor - fraction;
        }

        Self {
            fraction: T::try_from(fraction).unwrap_or_else(|_| {
                panic!("Failed to convert fraction to target type due to out-of-bounds value")
            }),
        }
    }
}

macro_rules! impl_safe_angle_conversions {
    ($($smaller:ty => $larger:ty),*) => {
        $(
            // Upscaling conversion (smaller to larger type) - Always safe
            impl From<Angle<$smaller>> for Angle<$larger> {
                fn from(angle: Angle<$smaller>) -> Self {
                    let shift = <$larger>::BITS - <$smaller>::BITS;
                    let scaled = <$larger>::from(angle.fraction) << shift;
                    Self { fraction: scaled }
                }
            }


            // Downscaling conversion (larger to smaller type) - Checked for safety
            impl TryFrom<Angle<$larger>> for Angle<$smaller> {
            type Error = &'static str;

            fn try_from(angle: Angle<$larger>) -> Result<Self, Self::Error> {
                let shift = <$larger>::BITS - <$smaller>::BITS;
                let mask = (1 << shift) - 1;

                if angle.fraction & mask != 0 {
                    return Err("Precision loss detected during angle conversion");
                }

                let shifted = angle.fraction >> shift;
                if let Ok(scaled) = <$smaller>::try_from(shifted) {
                    Ok(Self { fraction: scaled })
                } else {
                    Err("Value out of range for target type")
                }
            }
        }


        )*
    };
}

impl_safe_angle_conversions!(
    u8 => u16,
    u8 => u32,
    u8 => u64,
    u8 => u128,
    u16 => u32,
    u16 => u64,
    u16 => u128,
    u32 => u64,
    u32 => u128,
    u64 => u128
);

/// Macro to generate `LossyInto` implementations.
macro_rules! impl_lossy_into {
    ($($larger:ty => $smaller:ty),*$(,)?) => {
        $(
            impl LossyInto<Angle<$smaller>> for Angle<$larger> {
                #[allow(clippy::cast_possible_truncation)]
                fn lossy_into(self) -> Angle<$smaller> {
                    let mask = (1 << <$smaller>::BITS) - 1;
                    let scaled = self.fraction & mask;
                    Angle { fraction: scaled as $smaller }
                }
            }
        )*
    };
}

impl_lossy_into!(
    u16 => u8,
    u32 => u8,
    u64 => u8,
    u128 => u8,
    u32 => u16,
    u64 => u16,
    u128 => u16,
    u64 => u32,
    u128 => u32,
    u128 => u64
);

/// Trait for lossy conversions between types.
pub trait LossyInto<T>: Sized {
    fn lossy_into(self) -> T;
}

macro_rules! impl_angle_constants {
    ($t:ty) => {
        impl Angle<$t> {
            pub const ZERO: Self = Self { fraction: 0 };
            pub const QUARTER_TURN: Self = Self {
                fraction: 1 << (<$t>::BITS - 2),
            };
            pub const HALF_TURN: Self = Self {
                fraction: 1 << (<$t>::BITS - 1),
            };
            pub const THREE_QUARTERS_TURN: Self = Self {
                fraction: 3 << (<$t>::BITS - 2),
            };
            pub const FULL_TURN: Self = Self { fraction: 0 }; // Wraps to 0
        }
    };
}

impl_angle_constants!(u8);
impl_angle_constants!(u16);
impl_angle_constants!(u32);
impl_angle_constants!(u64);
impl_angle_constants!(u128);

/// Implements `From<f64>` for common Angle types, treating f64 as radians.
///
/// This allows convenient angle creation: `Angle64::from(1.5708)` or `angle.into()`
/// where the f64 value is interpreted as radians.
macro_rules! impl_from_f64 {
    ($($t:ty),*) => {
        $(
            impl From<f64> for Angle<$t> {
                fn from(radians: f64) -> Self {
                    Self::from_radians(radians)
                }
            }
        )*
    };
}

impl_from_f64!(u8, u16, u32, u64, u128);

/// Implements addition for angles, with modular wrapping.
impl<T> Add for Angle<T>
where
    T: Unsigned + WrappingAdd + Copy,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let sum = self.fraction.wrapping_add(&rhs.fraction);
        Self { fraction: sum }
    }
}

/// Implements addition assignment for angles, with modular wrapping.
impl<T: Unsigned + WrappingAdd + WrappingSub + Copy> AddAssign for Angle<T> {
    fn add_assign(&mut self, rhs: Self) {
        let sum = self.fraction.wrapping_add(&rhs.fraction);
        self.fraction = sum;
    }
}

/// Implements subtraction for angles, with modular wrapping.
impl<T: Unsigned + WrappingAdd + WrappingSub + Copy> Sub for Angle<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            fraction: self.fraction.wrapping_sub(&other.fraction),
        }
    }
}

/// Implements subtraction assignment for angles, with modular wrapping
impl<T: Unsigned + WrappingAdd + WrappingSub + Copy> SubAssign for Angle<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.fraction = self.fraction.wrapping_sub(&rhs.fraction);
    }
}

/// Implements scalar multiplication for angles.
impl<T: Unsigned + Copy + WrappingMul> Mul<T> for Angle<T> {
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self {
            fraction: self.fraction.wrapping_mul(&scalar),
        }
    }
}

/// Implements scalar multiplication assignment for angles.
impl<T: Unsigned + Copy + WrappingMul> MulAssign<T> for Angle<T> {
    fn mul_assign(&mut self, scalar: T) {
        self.fraction = self.fraction.wrapping_mul(&scalar);
    }
}

/// Implements scalar division for angles.
impl<T: Unsigned + Copy + FromPrimitive> Div<T> for Angle<T> {
    type Output = Self;

    fn div(self, scalar: T) -> Self {
        Self {
            fraction: self.fraction / scalar,
        }
    }
}

// Implement DivAssign for Angle
impl<T: Unsigned + Copy + FromPrimitive> DivAssign<T> for Angle<T> {
    fn div_assign(&mut self, scalar: T) {
        self.fraction = self.fraction / scalar;
    }
}

/// Implements negation for angles (computes the additive inverse modulo 2π).
impl<T: Unsigned + WrappingSub + Zero + Copy> Neg for Angle<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            fraction: T::zero().wrapping_sub(&self.fraction),
        }
    }
}

/// Implements `Display` for angles in terms of turns.
impl<T: Unsigned + ToPrimitive + Bounded + Copy> fmt::Display for Angle<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let fraction = self
            .fraction
            .to_f64()
            .expect("Failed to convert fraction to f64")
            / T::max_value()
                .to_f64()
                .expect("Failed to convert max_value to f64");
        write!(f, "{fraction:.6} turns")
    }
}

#[cfg(test)]
mod tests {
    use super::LossyInto;
    use super::*;
    use rand::RngExt;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

    // Basic Construction and Properties
    #[test]
    fn test_constructors() {
        let angle = Angle64::new(0);
        assert_eq!(angle, Angle64::ZERO);

        let normalized_angle = Angle64::from_radians(7.0 * PI).normalize();
        assert!((normalized_angle.to_radians() - PI).abs() < 1e-10);
    }

    #[test]
    fn test_zero_angle() {
        let zero = Angle64::ZERO;
        assert!(zero.is_zero());
        assert!((zero.to_radians()).abs() < 1e-10);
        assert!((zero.sin()).abs() < 1e-10);
        assert!((zero.cos() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_constant_relationships() {
        // Test relationships between constants
        assert_eq!(
            Angle64::HALF_TURN.fraction,
            Angle64::QUARTER_TURN.fraction * 2
        );
        assert_eq!(
            Angle64::THREE_QUARTERS_TURN.fraction,
            Angle64::QUARTER_TURN.fraction * 3
        );
        assert_eq!(Angle64::FULL_TURN.fraction, 0);

        // Test that constants maintain expected relationships in radians
        assert!((Angle64::QUARTER_TURN.to_radians() - FRAC_PI_2).abs() < 1e-10);
        assert!((Angle64::THREE_QUARTERS_TURN.to_radians() - (3.0 * FRAC_PI_2)).abs() < 1e-10);
    }

    // Basic Arithmetic Operations
    #[test]
    fn test_addition() {
        let a = Angle64::QUARTER_TURN;
        let b = Angle64::QUARTER_TURN;
        let result = a + b;
        assert_eq!(result.fraction, Angle64::HALF_TURN.fraction);
    }

    #[test]
    fn test_angle_arithmetic() {
        let quarter = Angle64::QUARTER_TURN;
        let half = Angle64::HALF_TURN;
        let three_quarters = Angle64::THREE_QUARTERS_TURN;
        let full = Angle64::FULL_TURN;

        assert_eq!((quarter + quarter).fraction, half.fraction);
        assert_eq!((quarter + half).fraction, three_quarters.fraction);
        assert_eq!((quarter + three_quarters).fraction, full.fraction);
    }

    #[test]
    fn test_u128_arithmetic() {
        let quarter = Angle128::QUARTER_TURN;
        let half = Angle128::HALF_TURN;

        // Test addition
        assert_eq!((quarter + quarter).fraction, half.fraction);

        // Test multiplication
        let doubled = quarter * 2u128;
        assert_eq!(doubled.fraction, half.fraction);

        // Test division
        let halved = half / 2u128;
        assert_eq!(halved.fraction, quarter.fraction);
    }

    #[test]
    fn test_division_edge_cases() {
        let angle = Angle64::HALF_TURN;

        // Test division by smallest value
        let divided = angle / 1u64;
        assert_eq!(divided, angle);

        // Test division by power of 2 (should be exact)
        let divided = angle / 4u64;
        assert_eq!(divided, Angle64::QUARTER_TURN / 2u64);

        // Test division of zero angle
        let zero = Angle64::ZERO / 1000u64;
        assert!(zero.is_zero());
    }

    #[should_panic(expected = "attempt to divide by zero")]
    #[test]
    fn test_division_by_zero() {
        let angle = Angle64::HALF_TURN;
        let _ = angle / 0u64;
    }

    #[test]
    fn test_scalar_operations() {
        let quarter = Angle64::QUARTER_TURN;
        let half = Angle64::HALF_TURN;

        // Test multiplication
        let doubled = quarter * 2u64;
        assert_eq!(doubled.fraction, half.fraction);

        // Test division
        let halved = half / 2u64;
        assert_eq!(halved.fraction, quarter.fraction);
    }

    #[test]
    fn test_large_scalar_division() {
        let angle = Angle64::QUARTER_TURN;

        // Test that division by larger and larger numbers produces smaller and smaller angles
        let small = angle / 1000u64;
        let tiny = angle / (u64::MAX / 2);
        assert!(small.to_radians() > tiny.to_radians());

        // Division by sufficiently large numbers will eventually produce zero
        // due to integer division behavior
        let microscopic = angle / u64::MAX;
        assert!(microscopic.to_radians() >= 0.0); // Should either be 0 or very small positive
    }

    #[test]
    fn test_scalar_multiplication_overflow() {
        let angle = Angle64::QUARTER_TURN;

        // This should wrap around due to overflow
        let result = angle * u64::MAX;
        assert_ne!(result.fraction, angle.fraction);

        // Multiplying by 4 should give us a full turn (0)
        let result = Angle64::QUARTER_TURN * 4u64;
        assert!(result.is_zero());
    }

    #[test]
    fn test_addition_subtraction_reversibility() {
        let quarter = Angle64::QUARTER_TURN;
        let half = Angle64::HALF_TURN;

        let test_angle = quarter + half;
        assert_eq!((test_angle - half).fraction, quarter.fraction);
    }

    #[test]
    fn test_add_assign() {
        let mut angle = Angle::<u64>::ZERO;
        angle += Angle::<u64>::QUARTER_TURN;
        assert_eq!(angle.fraction, Angle::<u64>::QUARTER_TURN.fraction);

        angle += Angle::<u64>::HALF_TURN;
        assert_eq!(angle.fraction, Angle::<u64>::THREE_QUARTERS_TURN.fraction);
    }

    #[test]
    fn test_sub_assign() {
        let mut angle = Angle::<u64>::HALF_TURN;
        angle -= Angle::<u64>::QUARTER_TURN;
        assert_eq!(angle.fraction, Angle::<u64>::QUARTER_TURN.fraction);
    }

    #[test]
    fn test_mul_assign() {
        let mut angle = Angle::<u64>::QUARTER_TURN;
        angle *= 2u64;
        assert_eq!(angle.fraction, Angle::<u64>::HALF_TURN.fraction);

        angle *= 2u64;
        assert!(angle.is_zero()); // FULL_TURN wraps to ZERO
    }

    #[test]
    fn test_div_assign() {
        let mut angle = Angle::<u64>::HALF_TURN; // A non-zero starting point
        angle /= 2u64; // Should result in a quarter turn
        assert_eq!(
            angle.fraction,
            Angle::<u64>::QUARTER_TURN.fraction,
            "not getting 1/2 turn / 2 ==  1/4 turn"
        );

        angle /= 2u64; // Should result in an eighth turn
        let expected = Angle::<u64>::from_turn_ratio(1, 8); // 1/8 of a turn
        assert_eq!(
            angle.fraction, expected.fraction,
            "not getting 1 / 4 turn / 2 ==  1/8 turn"
        );
    }

    #[should_panic(expected = "attempt to divide by zero")]
    #[test]
    fn test_div_assign_by_zero() {
        let mut angle = Angle64::HALF_TURN;
        angle /= 0u64;
    }

    #[test]
    fn test_assign_operations_chaining() {
        let mut angle: Angle<u64> = Angle::<u64>::QUARTER_TURN;
        angle *= 2u64;
        assert_eq!(
            angle.fraction,
            Angle::<u64>::HALF_TURN.fraction,
            "not getting 1/4 turn * 2 ==  1/2 turn"
        );

        angle /= 2u64;
        assert_eq!(
            angle.fraction,
            Angle::<u64>::QUARTER_TURN.fraction,
            "not getting 1/2 turn / 2 ==   1/4 turn"
        );
    }

    #[test]
    fn test_accumulation() {
        let quarter = Angle64::QUARTER_TURN;

        // Test accumulation of quarter turns
        let mut angle = Angle64::ZERO;
        for _ in 0..4 {
            angle += quarter;
        }
        assert_eq!(angle.fraction, Angle64::ZERO.fraction);

        // Test fine-grained accumulation
        let step = Angle64::QUARTER_TURN / 16u64;
        let accumulated = (0..16).fold(Angle64::ZERO, |acc, _| acc + step);
        assert_eq!(accumulated.fraction, Angle64::QUARTER_TURN.fraction);
    }

    #[test]
    fn test_precision_conversion() {
        let small_angle = Angle::<u128>::new(1);
        let converted: Result<Angle<u64>, _> = Angle::<u64>::try_from(small_angle);
        assert!(
            converted.is_err(),
            "Expected precision loss for small_angle"
        );
    }

    #[test]
    fn test_very_small_angles() {
        // Test smallest possible non-zero angle for each bit width
        let small32 = Angle32::new(1);
        let small64 = Angle64::new(1);
        let small128 = Angle128::new(1);

        assert!(!small32.is_zero());
        assert!(!small64.is_zero());
        assert!(!small128.is_zero());

        // These should all be very close to zero but not exactly zero
        assert!(small32.to_radians() > 0.0);
        assert!(small64.to_radians() > 0.0);
        assert!(small128.to_radians() > 0.0);
    }

    #[test]
    fn test_near_full_turn() {
        // Test angles very close to a full turn
        let almost_full32 = Angle32::new(u32::MAX);
        let almost_full64 = Angle64::new(u64::MAX);
        let almost_full128 = Angle128::new(u128::MAX);

        // Should all be very close to TAU but not exactly TAU
        assert!((almost_full32.to_radians() - TAU).abs() < 1e-6);
        assert!((almost_full64.to_radians() - TAU).abs() < 1e-10);
        assert!((almost_full128.to_radians() - TAU).abs() < 1e-10);

        // Adding 1 to these should wrap to 0
        assert!((almost_full64 + Angle64::new(1)).is_zero());
    }

    // Conversion and Representation
    #[test]
    fn test_to_radians() {
        let angle = Angle32 {
            fraction: u32::MAX / 2,
        };
        assert!((angle.to_radians() - PI).abs() < 1e-6);
    }

    #[test]
    fn test_from_radians_boundary() {
        use std::f64::consts::TAU;

        // Test exact TAU (should wrap to 0)
        let angle = Angle64::from_radians(TAU);
        assert!(angle.is_zero());

        // Test negative angles
        let angle = Angle64::from_radians(-PI);
        assert_eq!(angle.fraction, Angle64::HALF_TURN.fraction);

        // Test slightly negative angles
        let angle = Angle64::from_radians(-0.1);
        assert!((angle.to_radians() - (TAU - 0.1)).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Conversion of fraction to target type failed")]
    fn test_from_radians_overflow() {
        let _ = Angle32::from_radians(f64::INFINITY);
    }

    #[test]
    fn test_display() {
        let angle = Angle64 {
            fraction: u64::MAX / 4,
        };
        assert_eq!(format!("{angle}"), "0.250000 turns");
    }

    #[test]
    fn test_bit_width_conversions() {
        let angle32 = Angle32::QUARTER_TURN;
        let angle64: Angle64 = angle32.into();
        let back32: Angle32 = angle64.try_into().expect("Lossless conversion failed");
        assert_eq!(angle32, back32);

        let angle128: Angle128 = angle64.into();
        let back64: Angle64 = angle128.try_into().expect("Lossless conversion failed");
        assert_eq!(angle64, back64);
    }

    // Trigonometric Functions
    #[test]
    fn test_trig_functions() {
        let quarter = Angle64::QUARTER_TURN;
        assert!((quarter.sin() - 1.0).abs() < 1e-10);
        assert!(quarter.cos().abs() < 1e-10);

        let eighth = quarter / 2u64;
        assert!((eighth.sin() - FRAC_PI_4.sin()).abs() < 1e-10);
        assert!((eighth.cos() - FRAC_PI_4.cos()).abs() < 1e-10);

        let angle = Angle64::from_radians(FRAC_PI_4);
        assert!((angle.tan() - 1.0).abs() < 1e-10);
    }

    // Special Properties
    #[test]
    fn test_wrapping_behavior() {
        let quarter = Angle64::QUARTER_TURN;
        let full = Angle64::FULL_TURN;

        // Test wrapping around full circle
        let wrapped = full + quarter;
        assert_eq!(wrapped.fraction, quarter.fraction);

        // Test multiple wraps
        let mut angle = Angle64::ZERO;
        for _ in 0..8 {
            // Two full rotations
            angle += quarter;
        }
        assert_eq!(angle.fraction, Angle64::ZERO.fraction);
    }

    #[test]
    fn test_ordering() {
        let zero = Angle64::ZERO;
        let quarter = Angle64::QUARTER_TURN;
        let half = Angle64::HALF_TURN;

        assert!(zero < quarter);
        assert!(quarter < half);
        assert!(zero < half);

        let angles = vec![half, zero, quarter];
        let mut sorted = angles.clone();
        sorted.sort();
        assert_eq!(sorted, vec![zero, quarter, half]);
    }

    #[test]
    fn test_fraction_ordering() {
        let zero = Angle64::ZERO;
        let quarter = Angle64::QUARTER_TURN;
        let half = Angle64::HALF_TURN;
        let three_quarters = Angle64::THREE_QUARTERS_TURN;
        let full = Angle64::FULL_TURN;

        // Test fraction ordering
        assert!(quarter.fraction < half.fraction);
        assert!(half.fraction < three_quarters.fraction);
        assert!(three_quarters.fraction > quarter.fraction);

        // Test extremes
        assert_eq!(zero.fraction, 0);
        assert_eq!(full.fraction, 0); // Full turn wraps to 0

        // Test that fractions increase monotonically
        assert!(zero.fraction < quarter.fraction);
        assert!(quarter.fraction < half.fraction);
        assert!(half.fraction < three_quarters.fraction);
    }

    // Implementation Details/Edge Cases
    #[test]
    fn test_effective_modulus_u8() {
        let max_value = u8::MAX; // 255
        let effective_modulus = max_value.wrapping_add(1);
        assert_eq!(
            effective_modulus, 0,
            "Effective modulus should wrap to 0 for u8"
        );
    }

    #[test]
    fn test_effective_modulus_u64() {
        let max_value = u64::MAX;
        let effective_modulus = max_value.wrapping_add(1);
        assert_eq!(
            effective_modulus, 0,
            "Effective modulus should wrap to 0 for u64"
        );
    }

    #[test]
    fn test_angle_arithmetic_with_u16() {
        let a = Angle { fraction: 100_u16 };
        let b = Angle { fraction: 200_u16 };
        let result = a + b;
        assert!(result.fraction < u16::MAX, "Result must be within bounds");
    }

    #[test]
    fn test_angle_u32_to_u64_lossless() {
        let angle_u32 = Angle { fraction: 1_u32 };
        let angle_u64: Angle<u64> = angle_u32.into();
        assert_eq!(angle_u64.fraction, 1_u64 << 32);

        let angle_u32 = Angle { fraction: u32::MAX };
        let angle_u64: Angle<u64> = angle_u32.into();
        assert_eq!(angle_u64.fraction, u64::from(u32::MAX) << 32);
    }

    #[test]
    fn test_angle_u64_to_u32_lossy() {
        let angle = Angle::<u64>::new(1 << 33); // Value fits into u32 after shifting
        let result: Result<Angle<u32>, _> = Angle::try_from(angle);
        assert!(
            result.is_ok(),
            "Expected lossless conversion for 1 << 33, got {result:?}"
        );
    }

    #[test]
    fn test_angle_constants_conversion() {
        let zero_u32 = Angle { fraction: 0_u32 };
        let zero_u64: Angle<u64> = zero_u32.into();
        assert_eq!(zero_u64.fraction, 0_u64);
        assert_eq!(
            Angle::<u32>::try_from(zero_u64)
                .expect("Lossless conversion failed")
                .fraction,
            0_u32
        );

        let quarter_u32 = Angle {
            fraction: 1_u32 << 30,
        }; // 2^30
        let quarter_u64: Angle<u64> = quarter_u32.into();
        assert_eq!(quarter_u64.fraction, 1_u64 << 62);
        assert_eq!(
            Angle::<u32>::try_from(quarter_u64)
                .expect("Lossless conversion failed")
                .fraction,
            1_u32 << 30
        );

        let half_u32 = Angle {
            fraction: 1_u32 << 31,
        }; // 2^31
        let half_u64: Angle<u64> = half_u32.into();
        assert_eq!(half_u64.fraction, 1_u64 << 63);
        assert_eq!(
            Angle::<u32>::try_from(half_u64)
                .expect("Lossless conversion failed")
                .fraction,
            1_u32 << 31
        );

        let full_u32 = Angle { fraction: 0_u32 }; // Wraps to 0
        let full_u64: Angle<u64> = full_u32.into();
        assert_eq!(full_u64.fraction, 0_u64);
        assert_eq!(
            Angle::<u32>::try_from(full_u64)
                .expect("Lossless conversion failed")
                .fraction,
            0_u32
        );
    }

    #[test]
    fn test_round_trip_conversion_u32_u64() {
        let angle_u32 = Angle::<u32>::new(123_456);
        let converted: Angle<u64> = angle_u32.into();
        let back: Angle<u32> = converted.try_into().expect("Lossless conversion failed");
        assert_eq!(angle_u32, back);
    }

    #[test]
    fn test_round_trip_conversion_u64_u32() {
        let angle_u64 = Angle::<u64>::new(1 << 40);
        let converted: Angle<u32> = angle_u64.try_into().expect("Lossless conversion failed");
        let back: Angle<u64> = converted.into();
        // Check for approximate equality due to lossy conversion
        assert_eq!(back.fraction >> 32, angle_u64.fraction >> 32);
    }

    #[test]
    fn test_randomized_values_u32_to_u64() {
        let mut rng = rand::rng();
        for _ in 0..1000 {
            let random_u32: u32 = rng.random();
            let angle_u32 = Angle::<u32>::new(random_u32);
            let converted: Angle<u64> = angle_u32.into();
            let back: Angle<u32> = converted.try_into().expect("Lossless conversion failed");
            assert_eq!(angle_u32, back);
        }
    }

    #[test]
    fn test_randomized_values_u64_to_u32() {
        let mut rng = rand::rng();
        for _ in 0..1000 {
            let random_u64: u64 = rng.random();
            let angle_u64 = Angle::<u64>::new(random_u64);
            let result: Result<Angle<u32>, _> = angle_u64.try_into();
            if u32::try_from(random_u64).is_ok() {
                assert!(
                    result.is_ok(),
                    "Conversion should succeed for value within u32 range"
                );
            } else {
                assert!(
                    result.is_err(),
                    "Conversion should fail for value outside u32 range"
                );
            }
        }
    }

    #[test]
    fn test_subdivision_values() {
        // Test boundary cases for small subdivision values during conversions
        let angle_u32 = Angle::<u32>::new(1);
        let converted: Angle<u64> = angle_u32.into();
        assert_eq!(converted.fraction, 1_u64 << 32);

        let angle_u64 = Angle::<u64>::new(1);
        let converted: Result<Angle<u32>, _> = angle_u64.try_into();
        assert!(converted.is_err(), "Expected lossless conversion to fail");
    }

    #[test]
    fn test_near_boundary_values() {
        let angle_u32 = Angle::<u32>::new(u32::MAX - 1);
        let converted: Angle<u64> = angle_u32.into();
        let back: Result<Angle<u32>, _> = Angle::<u32>::try_from(converted);
        assert!(back.is_ok(), "Expected lossless conversion near boundary");

        let angle_u64 = Angle::<u64>::new((1 << 32) - 1);
        let back: Result<Angle<u32>, _> = Angle::<u32>::try_from(angle_u64);
        assert!(back.is_err(), "Expected lossless conversion to fail");
    }

    #[test]
    fn test_overflow_and_underflow() {
        let angle_u32 = Angle::<u32>::new(u32::MAX);
        let converted: Angle<u64> = angle_u32.into();
        let back: Result<Angle<u32>, _> = Angle::<u32>::try_from(converted);
        assert!(back.is_ok(), "Expected lossless conversion to succeed");

        let angle_u64 = Angle::<u64>::new(u64::MAX);
        let back: Result<Angle<u32>, _> = Angle::<u32>::try_from(angle_u64);
        assert!(back.is_err(), "Expected lossless conversion to fail");
    }

    #[test]
    fn test_non_uniform_scaling() {
        let angle_u32 = Angle::<u32>::new(u32::MAX / 3);
        let converted: Angle<u64> = angle_u32.into();
        let back: Result<Angle<u32>, _> = Angle::<u32>::try_from(converted);
        assert!(back.is_ok(), "Expected lossless conversion to succeed");

        let angle_u64 = Angle::<u64>::new(u64::MAX / 3);
        let back: Result<Angle<u32>, _> = Angle::<u32>::try_from(angle_u64);
        assert!(back.is_err(), "Expected lossless conversion to fail");
    }

    #[test]
    fn test_constants_conversion() {
        // Test that predefined constants are correctly converted between types
        assert_eq!(
            Angle::<u32>::try_from(Angle::<u64>::ZERO).expect("Lossless conversion failed"),
            Angle::<u32>::ZERO
        );
        assert_eq!(Angle::<u64>::from(Angle::<u32>::ZERO), Angle::<u64>::ZERO);

        assert_eq!(
            Angle::<u32>::try_from(Angle::<u64>::HALF_TURN).expect("Lossless conversion failed"),
            Angle::<u32>::HALF_TURN
        );
        assert_eq!(
            Angle::<u64>::from(Angle::<u32>::HALF_TURN),
            Angle::<u64>::HALF_TURN
        );

        assert_eq!(
            Angle::<u32>::try_from(Angle::<u64>::QUARTER_TURN).expect("Lossless conversion failed"),
            Angle::<u32>::QUARTER_TURN
        );
        assert_eq!(
            Angle::<u64>::from(Angle::<u32>::QUARTER_TURN),
            Angle::<u64>::QUARTER_TURN
        );

        assert_eq!(
            Angle::<u32>::try_from(Angle::<u64>::FULL_TURN).expect("Lossless conversion failed"),
            Angle::<u32>::FULL_TURN
        );
        assert_eq!(
            Angle::<u64>::from(Angle::<u32>::FULL_TURN),
            Angle::<u64>::FULL_TURN
        );
    }

    #[test]
    #[should_panic(expected = "attempt to divide by zero")]
    fn test_from_turn_ratio_panic_on_zero_denominator() {
        let _ = Angle64::from_turn_ratio(1, 0);
    }

    #[test]
    #[should_panic(
        expected = "Failed to convert fraction to target type due to out-of-bounds value"
    )]
    fn test_from_turn_ratio_panic_on_numerator_overflow() {
        let _ = Angle64::from_turn_ratio(i64::MAX, 1);
    }

    #[test]
    fn test_from_turn_ratio_valid_cases() {
        assert_eq!(
            Angle64::from_turn_ratio(1, 2).fraction,
            Angle64::HALF_TURN.fraction
        );
        assert_eq!(
            Angle64::from_turn_ratio(1, 4).fraction,
            Angle64::HALF_TURN.fraction / 2
        );
        assert_eq!(
            Angle64::from_turn_ratio(3, 4).fraction,
            3 * (Angle64::HALF_TURN.fraction / 2)
        );
    }

    #[test]
    fn test_from_turn_ratio_negative_numerator() {
        // -3/4 turn is equivalent to 1/4 turn
        let angle = Angle64::from_turn_ratio(-3, 4);
        assert_eq!(angle.fraction, Angle64::QUARTER_TURN.fraction);
    }

    #[test]
    fn test_from_turn_ratio_negative_denominator() {
        // 3/-4 turn is equivalent to -3/4 turn, which is 1/4 turn
        let angle = Angle64::from_turn_ratio(3, -4);
        assert_eq!(angle.fraction, Angle64::QUARTER_TURN.fraction);
    }

    #[test]
    fn test_from_turn_ratio_both_negative() {
        // -3/-4 turn is equivalent to 3/4 turn
        let angle = Angle64::from_turn_ratio(-3, -4);
        assert_eq!(angle.fraction, Angle64::THREE_QUARTERS_TURN.fraction);
    }

    #[test]
    fn test_from_turn_ratio_positive_case() {
        // Ensure positive values are handled correctly
        let angle = Angle64::from_turn_ratio(1, 2);
        assert_eq!(angle.fraction, Angle64::HALF_TURN.fraction);
    }

    #[test]
    fn test_from_turn_ratio_full_turn_negative() {
        // -1/1 turn wraps around to a full turn
        let angle = Angle64::from_turn_ratio(-1, 1);
        assert_eq!(angle.fraction, Angle64::FULL_TURN.fraction); // This is T::max_value()
    }

    #[test]
    fn test_from_turn_ratio_zero_numerator() {
        let angle = Angle64::from_turn_ratio(0, 1);
        assert_eq!(angle.fraction, Angle64::ZERO.fraction);
    }

    #[test]
    fn test_from_turn_ratio_valid_case() {
        let angle = Angle64::from_turn_ratio(1, 4);
        assert_eq!(angle.fraction, Angle64::HALF_TURN.fraction / 2);
    }

    #[test]
    fn test_from_turns_zero() {
        let angle = Angle::<u64>::from_turns(0.0);
        assert_eq!(angle.fraction, Angle::<u64>::ZERO.fraction);
    }

    #[test]
    fn test_from_turns_one_turn() {
        let angle = Angle::<u64>::from_turns(1.0);
        assert_eq!(angle.fraction, Angle::<u64>::ZERO.fraction); // 1 turn wraps back to 0
    }

    #[test]
    fn test_from_turns_half_turn() {
        let angle = Angle::<u64>::from_turns(0.5);
        assert_eq!(angle.fraction, Angle::<u64>::HALF_TURN.fraction); // Half a turn (180°)
    }

    #[test]
    fn test_from_turns_quarter_turn() {
        let angle = Angle::<u64>::from_turns(0.25);
        assert_eq!(angle.fraction, Angle::<u64>::HALF_TURN.fraction / 2); // Quarter turn (90°)
    }

    #[test]
    fn test_from_turns_negative_turn() {
        let angle = Angle::<u64>::from_turns(-0.25);
        assert_eq!(
            angle.fraction,
            Angle::<u64>::HALF_TURN.fraction + (Angle::<u64>::HALF_TURN.fraction / 2)
        ); // Negative quarter turn wraps to 3/4 turn
    }

    #[test]
    fn test_from_turns_large_turns() {
        let angle = Angle::<u64>::from_turns(5.75); // 5 full turns and 3/4 of a turn
        assert_eq!(
            angle.fraction,
            Angle::<u64>::HALF_TURN.fraction + (Angle::<u64>::HALF_TURN.fraction / 2)
        );
    }

    #[test]
    fn test_from_turns_small_fraction() {
        let angle = Angle::<u64>::from_turns(1e-9); // A very small fraction of a turn
        assert!(angle.fraction > 0); // Should not be zero
    }

    #[test]
    fn test_from_turns_negative_full_turn() {
        let angle = Angle::<u64>::from_turns(-1.0);
        assert_eq!(angle.fraction, Angle::<u64>::ZERO.fraction); // -1 turn wraps back to 0
    }

    #[test]
    fn test_from_turns_vs_from_radians_zero() {
        let angle_turns = Angle::<u64>::from_turns(0.0);
        let angle_radians = Angle::<u64>::from_radians(0.0);
        assert_eq!(angle_turns.fraction, angle_radians.fraction);
    }

    #[test]
    fn test_from_turns_vs_from_radians_full_turn() {
        let angle_turns = Angle::<u64>::from_turns(1.0);
        let angle_radians = Angle::<u64>::from_radians(std::f64::consts::TAU);
        assert_eq!(angle_turns.fraction, angle_radians.fraction);
    }

    #[test]
    fn test_from_turns_vs_from_radians_half_turn() {
        let angle_turns = Angle::<u64>::from_turns(0.5);
        let angle_radians = Angle::<u64>::from_radians(std::f64::consts::PI);
        assert_eq!(angle_turns.fraction, angle_radians.fraction);
    }

    #[test]
    fn test_from_turns_vs_from_radians_quarter_turn() {
        let angle_turns = Angle::<u64>::from_turns(0.25);
        let angle_radians = Angle::<u64>::from_radians(std::f64::consts::FRAC_PI_2);
        assert_eq!(angle_turns.fraction, angle_radians.fraction);
    }

    #[test]
    fn test_from_turns_vs_from_radians_negative_turn() {
        let angle_turns = Angle::<u64>::from_turns(-0.25);
        let angle_radians = Angle::<u64>::from_radians(-std::f64::consts::FRAC_PI_2);
        assert_eq!(angle_turns.fraction, angle_radians.fraction);
    }

    #[test]
    fn test_from_turns_vs_from_radians_random_values() {
        let random_turns = 0.123_456; // Example random value
        let random_radians = random_turns * std::f64::consts::TAU;

        let angle_turns = Angle::<u64>::from_turns(random_turns);
        let angle_radians = Angle::<u64>::from_radians(random_radians);

        // Allow a small tolerance for differences
        let tolerance = 256; // Increased allowable difference in the fraction
        assert!(
            (i128::from(angle_turns.fraction) - i128::from(angle_radians.fraction)).abs()
                <= tolerance,
            "angle_turns = {}, angle_radians = {}",
            angle_turns.fraction,
            angle_radians.fraction
        );
    }

    #[test]
    fn test_round_trip_conversion() {
        let values_u8 = [0, u8::MAX / 2, u8::MAX];
        let values_u16 = [0, u16::MAX / 2, u16::MAX];
        let values_u32 = [0, u32::MAX / 2, u32::MAX];
        let values_u64 = [0, u64::MAX / 2, u64::MAX];

        for &val in &values_u8 {
            let angle = Angle::<u8>::new(val);
            let up: Angle<u16> = angle.into();
            let down: Angle<u8> = up.try_into().expect("Lossless downscaling failed");
            assert_eq!(angle, down);
        }

        for &val in &values_u16 {
            let angle = Angle::<u16>::new(val);
            let up: Angle<u32> = angle.into();
            let down: Angle<u16> = up.try_into().expect("Lossless downscaling failed");
            assert_eq!(angle, down);
        }

        for &val in &values_u32 {
            let angle = Angle::<u32>::new(val);
            let up: Angle<u64> = angle.into();
            let down: Angle<u32> = up.try_into().expect("Lossless downscaling failed");
            assert_eq!(angle, down);
        }

        for &val in &values_u64 {
            let angle = Angle::<u64>::new(val);
            let up: Angle<u128> = angle.into();
            let down: Angle<u64> = up.try_into().expect("Lossless downscaling failed");
            assert_eq!(angle, down);
        }
    }

    #[test]
    fn test_randomized_conversions() {
        let mut rng = rand::rng();

        for _ in 0..1000 {
            let val_u8: u8 = rng.random();
            let angle = Angle::<u8>::new(val_u8);
            let up: Angle<u16> = angle.into();
            let down: Angle<u8> = up.try_into().expect("Lossless downscaling failed");
            assert_eq!(angle, down);

            let val_u16: u16 = rng.random();
            let angle = Angle::<u16>::new(val_u16);
            let up: Angle<u32> = angle.into();
            let down: Angle<u16> = up.try_into().expect("Lossless downscaling failed");
            assert_eq!(angle, down);

            let val_u32: u32 = rng.random();
            let angle = Angle::<u32>::new(val_u32);
            let up: Angle<u64> = angle.into();
            let down: Angle<u32> = up.try_into().expect("Lossless downscaling failed");
            assert_eq!(angle, down);

            let val_u64: u64 = rng.random();
            let angle = Angle::<u64>::new(val_u64);
            let up: Angle<u128> = angle.into();
            let down: Angle<u64> = up.try_into().expect("Lossless downscaling failed");
            assert_eq!(angle, down);
        }
    }

    #[test]
    fn test_predefined_constants_conversions() {
        assert_eq!(Angle::<u16>::from(Angle::<u8>::ZERO), Angle::<u16>::ZERO);
        assert_eq!(
            Angle::<u8>::try_from(Angle::<u16>::ZERO).unwrap(),
            Angle::<u8>::ZERO
        );

        assert_eq!(
            Angle::<u32>::from(Angle::<u16>::HALF_TURN),
            Angle::<u32>::HALF_TURN
        );
        assert_eq!(
            Angle::<u16>::try_from(Angle::<u32>::HALF_TURN).unwrap(),
            Angle::<u16>::HALF_TURN
        );

        assert_eq!(
            Angle::<u64>::from(Angle::<u32>::QUARTER_TURN),
            Angle::<u64>::QUARTER_TURN
        );
        assert_eq!(
            Angle::<u32>::try_from(Angle::<u64>::QUARTER_TURN).unwrap(),
            Angle::<u32>::QUARTER_TURN
        );

        assert_eq!(
            Angle::<u128>::from(Angle::<u64>::FULL_TURN),
            Angle::<u128>::FULL_TURN
        );
        assert_eq!(
            Angle::<u64>::try_from(Angle::<u128>::FULL_TURN).unwrap(),
            Angle::<u64>::FULL_TURN
        );
    }

    #[test]
    fn test_lossy_downscaling_conversions() {
        let angle = Angle::<u64>::new(1 << 63); // Value fits into u32 after shifting
        let result: Result<Angle<u32>, _> = Angle::try_from(angle);
        assert!(
            result.is_ok(),
            "Expected lossless conversion for 1 << 63, got {result:?}"
        );
    }

    #[test]
    fn test_upscaling_precision() {
        // Test that upscaling preserves precision
        let angle = Angle::<u8>::new(1);
        let up: Angle<u16> = angle.into();
        assert_eq!(up.fraction, 1 << 8);

        let angle = Angle::<u16>::new(1);
        let up: Angle<u32> = angle.into();
        assert_eq!(up.fraction, 1 << 16);

        let angle = Angle::<u32>::new(1);
        let up: Angle<u64> = angle.into();
        assert_eq!(up.fraction, 1 << 32);

        let angle = Angle::<u64>::new(1);
        let up: Angle<u128> = angle.into();
        assert_eq!(up.fraction, 1 << 64);
    }

    #[test]
    fn test_lossy_into_u64_to_u16() {
        let large_angle = Angle::<u64> { fraction: u64::MAX };
        let smaller_angle: Angle<u16> = large_angle.lossy_into();

        // The upper bits of `u64` are discarded, so we only expect the lower 16 bits
        assert_eq!(smaller_angle.fraction, u16::MAX);
    }

    #[test]
    fn test_lossy_into_u32_to_u8() {
        let large_angle = Angle::<u32> {
            fraction: 0x1234_ABCD,
        };
        let smaller_angle: Angle<u8> = large_angle.lossy_into();
        assert_eq!(smaller_angle.fraction, 0xCD);
    }

    #[test]
    fn test_lossy_into_and_back() {
        let original: Angle<u32> = Angle { fraction: 1024 };

        // Lossy conversion: u32 to u8
        let lossy: Angle<u8> = original.lossy_into();

        // Lossless conversion back: u8 to u32
        let back: Angle<u32> = lossy.into(); // Use `.into()` for lossless conversion

        // Verify that the lower bits are preserved during lossy conversion and back.
        let mask = (1 << 8) - 1; // Mask for the 8-bit range
        assert_eq!(original.fraction & mask, back.fraction);
    }

    #[test]
    fn test_lossy_into_zero_fraction() {
        let large_angle = Angle::<u64> { fraction: 0 };
        let smaller_angle: Angle<u16> = large_angle.lossy_into();

        // Zero should always remain zero regardless of conversion
        assert_eq!(smaller_angle.fraction, 0);
    }

    #[test]
    fn test_lossy_into_u128_to_u8() {
        let large_angle = Angle::<u128> {
            fraction: 0x1F2F_3F4F_5F6F_7F8F_FFFF_FFFF_FFFF_FFFF,
        };
        let smaller_angle: Angle<u8> = large_angle.lossy_into();
        assert_eq!(smaller_angle.fraction, 0xFF);
    }

    #[test]
    fn test_lossy_into_with_max_values() {
        let large_angle = Angle::<u64> { fraction: u64::MAX };
        let smaller_angle: Angle<u32> = large_angle.lossy_into();

        // Only the lower 32 bits should remain
        assert_eq!(smaller_angle.fraction, u32::MAX);
    }

    #[test]
    fn test_lossy_into_with_non_max_values() {
        let large_angle = Angle::<u32> {
            fraction: 0xFEDC_BA98,
        };
        let smaller_angle: Angle<u16> = large_angle.lossy_into();
        assert_eq!(smaller_angle.fraction, 0xBA98);
    }
}
