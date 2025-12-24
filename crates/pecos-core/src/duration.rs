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

//! Time duration types for quantum operations.
//!
//! This module provides [`Nanoseconds`], a typed wrapper around `u64` for representing
//! time durations in nanoseconds. This is useful for idle gates and timing-aware
//! circuit operations.
//!
//! # Example
//! ```rust
//! use pecos_core::Nanoseconds;
//!
//! let duration = Nanoseconds::from_ns(100);
//! assert_eq!(duration.as_ns(), 100);
//!
//! let duration_us = Nanoseconds::from_us(1);
//! assert_eq!(duration_us.as_ns(), 1000);
//! ```

use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// A time duration in nanoseconds.
///
/// This is a typed wrapper around `u64` that provides type safety and
/// convenience methods for working with time durations in quantum circuits.
///
/// # Example
/// ```rust
/// use pecos_core::Nanoseconds;
///
/// // Create from different time units
/// let ns = Nanoseconds::from_ns(100);
/// let us = Nanoseconds::from_us(1);  // 1000 ns
/// let ms = Nanoseconds::from_ms(1);  // 1_000_000 ns
///
/// // Arithmetic operations
/// let total = ns + us;
/// assert_eq!(total.as_ns(), 1100);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Nanoseconds(u64);

impl Nanoseconds {
    /// Zero duration.
    pub const ZERO: Self = Self(0);

    /// Create a duration from nanoseconds.
    #[must_use]
    pub const fn from_ns(ns: u64) -> Self {
        Self(ns)
    }

    /// Create a duration from microseconds.
    #[must_use]
    pub const fn from_us(us: u64) -> Self {
        Self(us * 1_000)
    }

    /// Create a duration from milliseconds.
    #[must_use]
    pub const fn from_ms(ms: u64) -> Self {
        Self(ms * 1_000_000)
    }

    /// Create a duration from seconds.
    #[must_use]
    pub const fn from_secs(secs: u64) -> Self {
        Self(secs * 1_000_000_000)
    }

    /// Get the duration in nanoseconds.
    #[must_use]
    pub const fn as_ns(self) -> u64 {
        self.0
    }

    /// Get the duration in microseconds (truncated).
    #[must_use]
    pub const fn as_us(self) -> u64 {
        self.0 / 1_000
    }

    /// Get the duration in milliseconds (truncated).
    #[must_use]
    pub const fn as_ms(self) -> u64 {
        self.0 / 1_000_000
    }

    /// Get the duration in seconds (truncated).
    #[must_use]
    pub const fn as_secs(self) -> u64 {
        self.0 / 1_000_000_000
    }

    /// Get the duration as f64 nanoseconds.
    ///
    /// Note: Precision loss may occur for durations > 2^53 ns (~104 days).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }
}

impl From<u64> for Nanoseconds {
    fn from(ns: u64) -> Self {
        Self(ns)
    }
}

impl From<Nanoseconds> for u64 {
    fn from(duration: Nanoseconds) -> Self {
        duration.0
    }
}

impl From<Nanoseconds> for f64 {
    #[allow(clippy::cast_precision_loss)]
    fn from(duration: Nanoseconds) -> Self {
        duration.0 as f64
    }
}

impl Add for Nanoseconds {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl AddAssign for Nanoseconds {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl Sub for Nanoseconds {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl SubAssign for Nanoseconds {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl Mul<u64> for Nanoseconds {
    type Output = Self;

    fn mul(self, rhs: u64) -> Self {
        Self(self.0 * rhs)
    }
}

impl MulAssign<u64> for Nanoseconds {
    fn mul_assign(&mut self, rhs: u64) {
        self.0 *= rhs;
    }
}

impl fmt::Display for Nanoseconds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ns", self.0)
    }
}

/// An abstract time duration in arbitrary units.
///
/// This is a typed wrapper around `u64` for representing time durations
/// when the actual time unit is unspecified or abstract (e.g., gate cycles,
/// time steps, or simulator ticks).
///
/// # Example
/// ```rust
/// use pecos_core::TimeUnits;
///
/// let duration = TimeUnits::new(100);
/// assert_eq!(duration.as_u64(), 100);
///
/// // Arithmetic operations
/// let total = duration + TimeUnits::new(50);
/// assert_eq!(total.as_u64(), 150);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TimeUnits(u64);

impl TimeUnits {
    /// Zero duration.
    pub const ZERO: Self = Self(0);

    /// Create a new time duration.
    #[must_use]
    pub const fn new(units: u64) -> Self {
        Self(units)
    }

    /// Get the duration as a u64.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Get the duration as f64.
    ///
    /// Note: Precision loss may occur for values > 2^53.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }
}

impl From<u64> for TimeUnits {
    fn from(units: u64) -> Self {
        Self(units)
    }
}

impl From<Nanoseconds> for TimeUnits {
    fn from(ns: Nanoseconds) -> Self {
        Self(ns.as_ns())
    }
}

impl From<TimeUnits> for u64 {
    fn from(duration: TimeUnits) -> Self {
        duration.0
    }
}

impl From<TimeUnits> for f64 {
    #[allow(clippy::cast_precision_loss)]
    fn from(duration: TimeUnits) -> Self {
        duration.0 as f64
    }
}

impl Add for TimeUnits {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl AddAssign for TimeUnits {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl Sub for TimeUnits {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl SubAssign for TimeUnits {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl Mul<u64> for TimeUnits {
    type Output = Self;

    fn mul(self, rhs: u64) -> Self {
        Self(self.0 * rhs)
    }
}

impl MulAssign<u64> for TimeUnits {
    fn mul_assign(&mut self, rhs: u64) {
        self.0 *= rhs;
    }
}

impl fmt::Display for TimeUnits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} units", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nanoseconds_conversions() {
        assert_eq!(Nanoseconds::from_ns(100).as_ns(), 100);
        assert_eq!(Nanoseconds::from_us(1).as_ns(), 1_000);
        assert_eq!(Nanoseconds::from_ms(1).as_ns(), 1_000_000);
        assert_eq!(Nanoseconds::from_secs(1).as_ns(), 1_000_000_000);
    }

    #[test]
    fn test_as_conversions() {
        let duration = Nanoseconds::from_ns(1_500_000);
        assert_eq!(duration.as_ns(), 1_500_000);
        assert_eq!(duration.as_us(), 1_500);
        assert_eq!(duration.as_ms(), 1);
        assert_eq!(duration.as_secs(), 0);
    }

    #[test]
    fn test_arithmetic() {
        let a = Nanoseconds::from_ns(100);
        let b = Nanoseconds::from_ns(50);

        assert_eq!((a + b).as_ns(), 150);
        assert_eq!((a - b).as_ns(), 50);
        assert_eq!((a * 3).as_ns(), 300);
    }

    #[test]
    fn test_from_u64() {
        let duration: Nanoseconds = 100u64.into();
        assert_eq!(duration.as_ns(), 100);
    }

    #[test]
    fn test_into_u64() {
        let duration = Nanoseconds::from_ns(100);
        let ns: u64 = duration.into();
        assert_eq!(ns, 100);
    }

    #[test]
    fn test_display() {
        let duration = Nanoseconds::from_ns(100);
        assert_eq!(format!("{duration}"), "100ns");
    }

    #[test]
    fn test_ordering() {
        let a = Nanoseconds::from_ns(100);
        let b = Nanoseconds::from_ns(200);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, Nanoseconds::from_ns(100));
    }

    // TimeUnits tests

    #[test]
    fn test_time_units_new() {
        let duration = TimeUnits::new(100);
        assert_eq!(duration.as_u64(), 100);
    }

    #[test]
    fn test_time_units_arithmetic() {
        let a = TimeUnits::new(100);
        let b = TimeUnits::new(50);

        assert_eq!((a + b).as_u64(), 150);
        assert_eq!((a - b).as_u64(), 50);
        assert_eq!((a * 3).as_u64(), 300);
    }

    #[test]
    fn test_time_units_from_u64() {
        let duration: TimeUnits = 100u64.into();
        assert_eq!(duration.as_u64(), 100);
    }

    #[test]
    fn test_time_units_display() {
        let duration = TimeUnits::new(100);
        assert_eq!(format!("{duration}"), "100 units");
    }
}
