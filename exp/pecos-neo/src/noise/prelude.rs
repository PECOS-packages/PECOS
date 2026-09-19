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

//! Unified prelude for the noise system.
//!
//! Re-exports everything needed to build noise models: pre-built patterns,
//! builder API, topology helpers, and validation. Composite primitives require
//! the `composite-noise` Cargo feature.
//!
//! For a full guide with examples, see `docs/experimental/composable-noise.md`.

// --- Unified Builder ---

pub use super::builder::NoiseModelBuilder;

// --- Composable Model ---

pub use super::composer::ComposableNoiseModel;

// --- Base Channels (Pre-built) ---

pub use super::correlated::CorrelatedNoiseChannel;
pub use super::crosstalk::CrosstalkChannel;
pub use super::idle::IdleChannel;
pub use super::leakage::LeakageChannel;
pub use super::measurement::MeasurementChannel;
pub use super::preparation::PreparationChannel;
pub use super::single_qubit::SingleQubitChannel;
pub use super::two_qubit::{AngleScaling, TwoQubitChannel};

// Re-export weights and transitions from the main noise module
pub use super::CrosstalkTransitions;
pub use super::PauliWeights;
pub use super::SingleQubitEmissionWeights;
pub use super::TwoQubitEmissionWeights;
pub use super::TwoQubitPauliWeights;

// --- Composite System (Composition) ---

#[cfg(feature = "composite-noise")]
pub use super::composite::prelude::*;

// --- Core Traits and Types ---

pub use super::{EventKinds, NoiseChannel, NoiseEvent, NoiseEventKind, NoiseResponse};
pub use super::{GateInfo, IdleInfo, NoiseContext};

// --- Topology (Spatial Noise Helpers) ---

pub use super::topology::{
    chain_distance, chain_neighbors, exponential_decay, gaussian_decay, grid_distance,
    grid_neighbors, power_law_decay,
};

// --- Convenience Patterns (Pre-built Configurations) ---

#[cfg(feature = "composite-noise")]
pub use super::patterns::{
    DeviceNoiseParams, chain_measurement_crosstalk, dephasing_only, grid_measurement_crosstalk,
    realistic_device_noise, surface_code_noise, with_leakage,
};
pub use super::patterns::{
    chain_correlated, depolarizing_only, depolarizing_with_measurement, measurement_only,
};

// --- Validation ---

pub use super::validation::{
    ValidationError, ValidationResult, clamp_probability, is_probability_one, is_probability_zero,
    validate_probability, validate_rate, validate_weights,
};
