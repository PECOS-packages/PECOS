// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Typed signals for inline metadata in command streams.
//!
//! Signals are user-defined types that flow alongside gate commands in a
//! [`CommandQueue`]. They carry metadata (zone temperatures, QEC round
//! boundaries, calibration data, etc.) that pipeline consumers can
//! subscribe to and interpret.
//!
//! # Defining a signal
//!
//! Most signals are simple newtypes wrapping a single value:
//!
//! ```
//! use pecos_core::impl_signal;
//!
//! #[derive(Copy, Clone, Debug)]
//! struct ZoneTemperature(pub f64);
//! impl_signal!(ZoneTemperature);
//! ```
//!
//! For complex data, use a struct with named fields:
//!
//! ```
//! use pecos_core::impl_signal;
//!
//! #[derive(Clone, Debug)]
//! struct CalibrationData {
//!     pub rates: [f64; 8],
//!     pub num_qubits: u8,
//! }
//! impl_signal!(CalibrationData);
//! ```

use std::any::Any;

/// Marker trait for types that can be sent as signals in a command stream.
///
/// A signal is a typed piece of data that flows alongside gate commands,
/// maintaining its position relative to those commands. The Rust type
/// acts as the routing key -- consumers subscribe to specific signal
/// types rather than matching on a generic enum.
///
/// Use [`impl_signal!`] to implement this trait:
///
/// ```
/// use pecos_core::impl_signal;
///
/// #[derive(Copy, Clone, Debug)]
/// struct RoundBoundary(pub i64);
/// impl_signal!(RoundBoundary);
/// ```
///
/// # Trait bounds
///
/// - `Any + 'static`: Enables type-erased storage with typed retrieval.
/// - `Send + Sync`: Signals may cross thread boundaries in parallel execution.
/// - `Clone`: Signals may be delivered to multiple consumers.
pub trait Signal: Any + Send + Sync + Clone + 'static {
    /// Human-readable name for debugging and diagnostics.
    fn name() -> &'static str;
}

/// Implement [`Signal`] for a type.
///
/// # Example
///
/// ```
/// use pecos_core::impl_signal;
///
/// #[derive(Copy, Clone, Debug)]
/// struct ZoneTemperature(pub f64);
/// impl_signal!(ZoneTemperature);
///
/// assert_eq!(<ZoneTemperature as pecos_core::Signal>::name(), "ZoneTemperature");
/// ```
#[macro_export]
macro_rules! impl_signal {
    ($t:ty) => {
        impl $crate::Signal for $t {
            fn name() -> &'static str {
                stringify!($t)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Copy, Clone, Debug, PartialEq)]
    struct Temperature(pub f64);
    impl_signal!(Temperature);

    #[derive(Copy, Clone, Debug, PartialEq)]
    struct RoundBoundary(pub i64);
    impl_signal!(RoundBoundary);

    #[derive(Clone, Debug, PartialEq)]
    struct CalibrationData {
        pub rates: [f64; 4],
        pub label: &'static str,
    }
    impl_signal!(CalibrationData);

    #[test]
    fn signal_name() {
        assert_eq!(Temperature::name(), "Temperature");
        assert_eq!(RoundBoundary::name(), "RoundBoundary");
        assert_eq!(CalibrationData::name(), "CalibrationData");
    }

    #[test]
    fn signal_is_any() {
        let temp = Temperature(300.0);
        let any_ref: &dyn Any = &temp;
        assert!(any_ref.downcast_ref::<Temperature>().is_some());
        let t = any_ref
            .downcast_ref::<Temperature>()
            .expect("downcast to Temperature");
        assert!((t.0 - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn signal_is_clone() {
        let cal = CalibrationData {
            rates: [0.01, 0.02, 0.03, 0.04],
            label: "test",
        };
        let cloned = cal.clone();
        assert_eq!(cal, cloned);
    }

    #[test]
    fn signal_type_id_distinguishes_types() {
        use std::any::TypeId;
        assert_ne!(TypeId::of::<Temperature>(), TypeId::of::<RoundBoundary>());
    }
}
