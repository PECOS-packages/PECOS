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
//! This module provides:
//!
//! - [`TimeUnits`]: Abstract time duration used by the underlying simulation system.
//!   This is a simple `u64` wrapper - the interpretation is left to the user.
//!
//! - [`TimeScale`]: Interprets `TimeUnits` as physical time. Provides convenience
//!   methods for converting to/from seconds, nanoseconds, `std::time::Duration`, etc.
//!
//! # Design Philosophy
//!
//! The simulation system works with abstract `TimeUnits` (like game engines use
//! abstract ticks). Users choose how to interpret these units via `TimeScale`.
//! This keeps the core simple while supporting any time scale.
//!
//! # Example
//! ```rust
//! use pecos_core::{TimeUnits, TimeScale};
//!
//! // Simulation works with abstract units
//! let idle_time = TimeUnits::new(100);
//!
//! // User interprets as nanoseconds
//! let scale = TimeScale::NANOSECONDS;
//! assert!((scale.to_seconds(idle_time) - 100e-9).abs() < 1e-18);
//!
//! // Or as microseconds
//! let scale = TimeScale::MICROSECONDS;
//! assert!((scale.to_seconds(idle_time) - 100e-6).abs() < 1e-15);
//!
//! // Or as gate cycles (e.g., 50ns per cycle)
//! let scale = TimeScale::from_cycle_time_ns(50.0);
//! assert!((scale.to_ns(idle_time) - 5000.0).abs() < 1e-6);
//! ```

use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

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

/// Interprets [`TimeUnits`] as physical time.
///
/// `TimeUnits` are abstract - this struct defines how to interpret them.
/// The underlying simulation always works in `TimeUnits`, but `TimeScale`
/// provides user-friendly conversion to/from physical time.
///
/// # Example
/// ```rust
/// use pecos_core::{TimeUnits, TimeScale};
///
/// // Define interpretation: 1 TimeUnit = 1 nanosecond
/// let scale = TimeScale::NANOSECONDS;
///
/// // Convert from physical time to TimeUnits
/// let units = scale.from_ns(100.0);
/// assert_eq!(units.as_u64(), 100);
///
/// // Convert from TimeUnits to physical time
/// assert!((scale.to_ns(units) - 100.0).abs() < 1e-10);
/// assert!((scale.to_seconds(units) - 100e-9).abs() < 1e-18);
///
/// // Gate cycle interpretation (e.g., 50ns per cycle)
/// let cycle_scale = TimeScale::from_cycle_time_ns(50.0);
/// let units = cycle_scale.from_ns(200.0);  // 200ns = 4 cycles
/// assert_eq!(units.as_u64(), 4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeScale {
    /// Seconds per `TimeUnit`.
    seconds_per_unit: f64,
}

impl TimeScale {
    /// 1 `TimeUnit` = 1 nanosecond.
    pub const NANOSECONDS: Self = Self {
        seconds_per_unit: 1e-9,
    };

    /// 1 `TimeUnit` = 1 microsecond.
    pub const MICROSECONDS: Self = Self {
        seconds_per_unit: 1e-6,
    };

    /// 1 `TimeUnit` = 1 millisecond.
    pub const MILLISECONDS: Self = Self {
        seconds_per_unit: 1e-3,
    };

    /// 1 `TimeUnit` = 1 second.
    pub const SECONDS: Self = Self {
        seconds_per_unit: 1.0,
    };

    /// Create a time scale from gate cycle duration in nanoseconds.
    ///
    /// # Example
    /// ```rust
    /// use pecos_core::{TimeUnits, TimeScale};
    ///
    /// // Each TimeUnit represents a 50ns gate cycle
    /// let scale = TimeScale::from_cycle_time_ns(50.0);
    ///
    /// // 10 cycles = 500ns
    /// assert!((scale.to_ns(TimeUnits::new(10)) - 500.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn from_cycle_time_ns(ns_per_cycle: f64) -> Self {
        Self {
            seconds_per_unit: ns_per_cycle * 1e-9,
        }
    }

    /// Create a time scale with custom seconds per unit.
    #[must_use]
    pub const fn from_seconds_per_unit(seconds: f64) -> Self {
        Self {
            seconds_per_unit: seconds,
        }
    }

    /// Add decimal precision to this time scale.
    ///
    /// This lets you think in a coarse unit (like seconds) while having
    /// fine-grained precision (like nanoseconds) in the underlying `TimeUnits`.
    ///
    /// # Arguments
    /// * `digits` - Number of decimal places of precision beyond the base unit
    ///
    /// # Example
    /// ```rust
    /// use pecos_core::{TimeUnits, TimeScale};
    ///
    /// // Think in seconds, but with nanosecond precision (9 decimal places)
    /// let scale = TimeScale::SECONDS.with_precision(9);
    ///
    /// // 0.00005 seconds (50 microseconds) = 50,000 TimeUnits
    /// let units = scale.from_seconds(0.00005);
    /// assert_eq!(units.as_u64(), 50_000);
    ///
    /// // This is equivalent to TimeScale::NANOSECONDS
    /// assert!((scale.seconds_per_unit() - 1e-9).abs() < 1e-18);
    /// ```
    #[must_use]
    pub fn with_precision(self, digits: u8) -> Self {
        let factor = 10f64.powi(i32::from(digits));
        Self {
            seconds_per_unit: self.seconds_per_unit / factor,
        }
    }

    /// Get the seconds per `TimeUnit` for this scale.
    #[must_use]
    pub const fn seconds_per_unit(&self) -> f64 {
        self.seconds_per_unit
    }

    // --- Conversion FROM physical time TO `TimeUnits` ---

    /// Convert seconds to `TimeUnits`.
    ///
    /// Rounds to nearest integer.
    #[must_use]
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    pub fn from_seconds(&self, seconds: f64) -> TimeUnits {
        let units = (seconds / self.seconds_per_unit).round();
        TimeUnits::new(units.max(0.0) as u64)
    }

    /// Convert nanoseconds to `TimeUnits`.
    ///
    /// Rounds to nearest integer.
    #[must_use]
    pub fn from_ns(&self, ns: f64) -> TimeUnits {
        self.from_seconds(ns * 1e-9)
    }

    /// Convert microseconds to `TimeUnits`.
    ///
    /// Rounds to nearest integer.
    #[must_use]
    pub fn from_us(&self, us: f64) -> TimeUnits {
        self.from_seconds(us * 1e-6)
    }

    /// Convert milliseconds to `TimeUnits`.
    ///
    /// Rounds to nearest integer.
    #[must_use]
    pub fn from_ms(&self, ms: f64) -> TimeUnits {
        self.from_seconds(ms * 1e-3)
    }

    /// Convert a `std::time::Duration` to `TimeUnits`.
    ///
    /// Rounds to nearest integer.
    #[must_use]
    pub fn from_duration(&self, duration: std::time::Duration) -> TimeUnits {
        self.from_seconds(duration.as_secs_f64())
    }

    // --- Conversion FROM `TimeUnits` TO physical time ---

    /// Convert `TimeUnits` to seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_seconds(&self, units: TimeUnits) -> f64 {
        units.0 as f64 * self.seconds_per_unit
    }

    /// Convert `TimeUnits` to nanoseconds.
    #[must_use]
    pub fn to_ns(&self, units: TimeUnits) -> f64 {
        self.to_seconds(units) * 1e9
    }

    /// Convert `TimeUnits` to microseconds.
    #[must_use]
    pub fn to_us(&self, units: TimeUnits) -> f64 {
        self.to_seconds(units) * 1e6
    }

    /// Convert `TimeUnits` to milliseconds.
    #[must_use]
    pub fn to_ms(&self, units: TimeUnits) -> f64 {
        self.to_seconds(units) * 1e3
    }

    /// Convert `TimeUnits` to a `std::time::Duration`.
    #[must_use]
    pub fn to_duration(&self, units: TimeUnits) -> std::time::Duration {
        std::time::Duration::from_secs_f64(self.to_seconds(units))
    }
}

impl Default for TimeScale {
    /// Default interpretation: 1 `TimeUnit` = 1 nanosecond.
    fn default() -> Self {
        Self::NANOSECONDS
    }
}

impl fmt::Display for TimeScale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if (self.seconds_per_unit - 1e-9).abs() < 1e-15 {
            write!(f, "1 unit = 1 ns")
        } else if (self.seconds_per_unit - 1e-6).abs() < 1e-12 {
            write!(f, "1 unit = 1 us")
        } else if (self.seconds_per_unit - 1e-3).abs() < 1e-9 {
            write!(f, "1 unit = 1 ms")
        } else if (self.seconds_per_unit - 1.0).abs() < 1e-6 {
            write!(f, "1 unit = 1 s")
        } else {
            // Use scientific notation for clarity
            write!(f, "1 unit = {:.3e} s", self.seconds_per_unit)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // TimeScale tests

    #[test]
    fn test_time_scale_nanoseconds() {
        let scale = TimeScale::NANOSECONDS;

        // 100 TimeUnits = 100 ns
        let units = TimeUnits::new(100);
        assert!((scale.to_ns(units) - 100.0).abs() < 1e-10);
        assert!((scale.to_seconds(units) - 100e-9).abs() < 1e-18);

        // 100 ns = 100 TimeUnits
        let units = scale.from_ns(100.0);
        assert_eq!(units.as_u64(), 100);
    }

    #[test]
    fn test_time_scale_microseconds() {
        let scale = TimeScale::MICROSECONDS;

        // 100 TimeUnits = 100 us
        let units = TimeUnits::new(100);
        assert!((scale.to_us(units) - 100.0).abs() < 1e-10);
        assert!((scale.to_ns(units) - 100_000.0).abs() < 1e-6);

        // 100 us = 100 TimeUnits
        let units = scale.from_us(100.0);
        assert_eq!(units.as_u64(), 100);
    }

    #[test]
    fn test_time_scale_gate_cycles() {
        // 50ns per gate cycle
        let scale = TimeScale::from_cycle_time_ns(50.0);

        // 10 cycles = 500 ns
        let units = TimeUnits::new(10);
        assert!((scale.to_ns(units) - 500.0).abs() < 1e-10);

        // 200 ns = 4 cycles
        let units = scale.from_ns(200.0);
        assert_eq!(units.as_u64(), 4);
    }

    #[test]
    fn test_time_scale_roundtrip() {
        let scale = TimeScale::NANOSECONDS;

        // Round-trip: ns -> TimeUnits -> ns
        let original_ns = 12345.0;
        let units = scale.from_ns(original_ns);
        let recovered_ns = scale.to_ns(units);
        assert!((recovered_ns - original_ns).abs() < 1.0); // Within 1 ns due to rounding
    }

    #[test]
    fn test_time_scale_duration_conversion() {
        let scale = TimeScale::MILLISECONDS;

        // std::time::Duration -> TimeUnits
        let duration = std::time::Duration::from_millis(100);
        let units = scale.from_duration(duration);
        assert_eq!(units.as_u64(), 100);

        // TimeUnits -> std::time::Duration
        let recovered = scale.to_duration(units);
        assert_eq!(recovered.as_millis(), 100);
    }

    #[test]
    fn test_time_scale_display() {
        assert_eq!(format!("{}", TimeScale::NANOSECONDS), "1 unit = 1 ns");
        assert_eq!(format!("{}", TimeScale::MICROSECONDS), "1 unit = 1 us");
        assert_eq!(format!("{}", TimeScale::MILLISECONDS), "1 unit = 1 ms");
        assert_eq!(format!("{}", TimeScale::SECONDS), "1 unit = 1 s");
        // Custom scale uses scientific notation
        assert_eq!(
            format!("{}", TimeScale::from_cycle_time_ns(50.0)),
            "1 unit = 5.000e-8 s"
        );
    }

    #[test]
    fn test_time_scale_default() {
        let scale = TimeScale::default();
        assert_eq!(scale, TimeScale::NANOSECONDS);
    }

    #[test]
    fn test_time_scale_with_precision() {
        // Seconds with 9 digits precision = nanoseconds
        let scale = TimeScale::SECONDS.with_precision(9);
        assert!((scale.seconds_per_unit() - 1e-9).abs() < 1e-18);

        // 50 microseconds = 0.00005 seconds = 50,000 nanoseconds
        let units = scale.from_seconds(0.00005);
        assert_eq!(units.as_u64(), 50_000);

        // Seconds with 6 digits precision = microseconds
        let scale = TimeScale::SECONDS.with_precision(6);
        assert!((scale.seconds_per_unit() - 1e-6).abs() < 1e-15);

        // 50 microseconds = 50 TimeUnits
        let units = scale.from_seconds(0.00005);
        assert_eq!(units.as_u64(), 50);

        // Milliseconds with 3 digits precision = microseconds
        let scale = TimeScale::MILLISECONDS.with_precision(3);
        assert!((scale.seconds_per_unit() - 1e-6).abs() < 1e-15);
    }
}
