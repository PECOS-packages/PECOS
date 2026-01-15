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

//! Macros for creating exact angles without floating-point arithmetic.
//!
//! These macros provide compile-time construction of [`Angle64`](crate::Angle64) values
//! using exact rational arithmetic, avoiding any floating-point precision loss.

/// Creates an exact `Angle64` from a pi-based expression.
///
/// This macro converts expressions involving pi to exact `Angle64` values
/// without any floating-point arithmetic. The conversion uses the fact that
/// `k * pi / n` radians equals `k / (2n)` turns.
///
/// # Supported Syntax
///
/// - `angle!(pi)` - π radians (half turn)
/// - `angle!(pi / n)` - π/n radians
/// - `angle!(k * pi)` - k*π radians
/// - `angle!(k * pi / n)` - k*π/n radians
/// - `angle!(-pi)`, `angle!(-pi / n)`, etc. - negative angles
///
/// # Examples
///
/// ```
/// use pecos_core::{angle, Angle64};
///
/// let quarter_turn = angle!(pi / 2);
/// assert_eq!(quarter_turn, Angle64::QUARTER_TURN);
///
/// let eighth_turn = angle!(pi / 4);
/// assert_eq!(eighth_turn, Angle64::HALF_TURN / 4);
///
/// let third_turn = angle!(2 * pi / 3);
/// ```
#[macro_export]
macro_rules! angle {
    // pi -> 1/2 turn
    (pi) => {
        $crate::Angle64::HALF_TURN
    };

    // -pi -> -1/2 turn
    (-pi) => {
        $crate::Angle64::ZERO - $crate::Angle64::HALF_TURN
    };

    // pi / n -> 1/(2n) turn
    (pi / $denom:tt) => {
        $crate::Angle64::from_turn_ratio(1, 2 * $denom)
    };

    // -pi / n -> -1/(2n) turn
    (-pi / $denom:tt) => {
        $crate::Angle64::from_turn_ratio(-1_i64, 2 * $denom)
    };

    // k * pi -> k/2 turn
    ($numer:tt * pi) => {
        $crate::Angle64::from_turn_ratio($numer, 2)
    };

    // k * pi / n -> k/(2n) turn
    ($numer:tt * pi / $denom:tt) => {
        $crate::Angle64::from_turn_ratio($numer, 2 * $denom)
    };
}

/// Creates an exact `Angle64` from a turn-based fraction.
///
/// This macro works directly in turns (fractions of a full rotation),
/// which is the native representation of `Angle64`. This is often more
/// intuitive for quantum computing where gates are commonly described
/// in terms of turns (e.g., T gate = 1/8 turn, S gate = 1/4 turn).
///
/// # Supported Syntax
///
/// - `turn!(1 / n)` - 1/n of a full turn
/// - `turn!(k / n)` - k/n of a full turn
/// - `turn!(-1 / n)`, `turn!(-k / n)` - negative turns
///
/// # Examples
///
/// ```
/// use pecos_core::{turn, Angle64};
///
/// // Common quantum gates in turns
/// let t_angle = turn!(1 / 8);      // T gate phase (π/4 radians)
/// let s_angle = turn!(1 / 4);      // S gate phase (π/2 radians)
/// let z_angle = turn!(1 / 2);      // Z gate phase (π radians)
///
/// assert_eq!(s_angle, Angle64::QUARTER_TURN);
/// assert_eq!(z_angle, Angle64::HALF_TURN);
///
/// // Arbitrary fractions
/// let third = turn!(1 / 3);        // 120 degrees
/// let two_thirds = turn!(2 / 3);   // 240 degrees
/// ```
#[macro_export]
macro_rules! turn {
    // k / n -> k/n turn
    ($numer:tt / $denom:tt) => {
        $crate::Angle64::from_turn_ratio($numer, $denom)
    };

    // -k / n -> -k/n turn
    (-$numer:tt / $denom:tt) => {
        $crate::Angle64::from_turn_ratio(-($numer as i64), $denom)
    };
}
