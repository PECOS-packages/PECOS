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

//! Built-in plugins for common noise model functionality.
//!
//! ## Core Plugins
//!
//! - [`CorePlugin`]: Fundamental state tracking (preparation, measurement)
//! - [`LeakagePlugin`]: Leakage effects (gate skipping, measurement handling)
//!
//! ## Noise Plugins
//!
//! - [`DepolarizingPlugin`]: Single and two-qubit depolarizing noise
//! - [`MeasurementNoisePlugin`]: Asymmetric measurement errors

mod core;
mod depolarizing;
mod leakage;
mod measurement;

pub use self::core::CorePlugin;
pub use self::depolarizing::DepolarizingPlugin;
pub use self::leakage::LeakagePlugin;
pub use self::measurement::MeasurementNoisePlugin;
