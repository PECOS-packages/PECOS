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
//! This module re-exports everything you need to build noise models, from simple
//! parameter-based configuration to complex composed decision trees.
//!
//! # Quick Start
//!
//! ```
//! use pecos_neo::noise::prelude::*;
//!
//! // Simple: just set error rates
//! let model = depolarizing_with_measurement(0.001, 0.01, 0.02);
//!
//! // Or use the builder for more control
//! let model = NoiseModelBuilder::new()
//!     .with_depolarizing(0.001, 0.01)
//!     .with_measurement_error(0.02)
//!     .build();
//! ```
//!
//! # Choosing Your Approach
//!
//! The noise system offers three levels of abstraction:
//!
//! ## 1. Pre-built Patterns (Easiest)
//!
//! Use ready-made noise configurations for common scenarios:
//!
//! ```
//! use pecos_neo::noise::prelude::*;
//!
//! // Simple depolarizing noise
//! let model = depolarizing_only(0.001, 0.01);
//!
//! // With measurement errors
//! let model = depolarizing_with_measurement(0.001, 0.01, 0.02);
//!
//! // Realistic device noise with all parameters
//! let model = realistic_device_noise(
//!     DeviceNoiseParams::new()
//!         .with_p1(0.001)
//!         .with_p2(0.01)
//!         .with_measurement_error(0.02)
//!         .with_t1(0.0001)
//! );
//!
//! // Surface code optimized noise
//! let model = surface_code_noise(0.001, true);
//! ```
//!
//! ## 2. Builder API (Flexible)
//!
//! Use `NoiseModelBuilder` to combine features:
//!
//! ```
//! use pecos_neo::noise::prelude::*;
//!
//! let model = NoiseModelBuilder::new()
//!     .with_depolarizing(0.001, 0.01)
//!     .with_measurement_error(0.02)
//!     .with_preparation_error(0.001)
//!     .with_idle_noise(0.0001, 0.0005)  // T1 and T2 rates
//!     .build();
//! ```
//!
//! ## 3. Composed Primitives (Full Control)
//!
//! Build custom decision trees using composite primitives:
//!
//! ```
//! use pecos_neo::noise::prelude::*;
//!
//! // Custom single-qubit noise with leakage handling
//! let sq_noise = seq![
//!     skip_if_leaked(),                    // Skip gate if qubit is leaked
//!     prob(0.001,                          // 0.1% error probability
//!         when_leaked(
//!             seep(),                      // Leaked: return to computational basis
//!             sample![
//!                 (0.1, leak()),           // 10% of errors cause leakage
//!                 (0.9, pauli()),          // 90% are Pauli errors
//!             ]
//!         )
//!     ),
//! ];
//!
//! let model = NoiseModelBuilder::new()
//!     .with_single_qubit_noise(sq_noise)
//!     .build();
//! ```
//!
//! # Common Recipes
//!
//! ## Depolarizing Noise
//!
//! Random Pauli errors after gates:
//!
//! ```
//! # use pecos_neo::noise::prelude::*;
//! // Using pattern
//! let model = depolarizing_only(0.001, 0.01);
//!
//! // Using primitives
//! let sq = CompositeChannelBuilder::single_qubit("sq", prob(0.001, pauli()));
//! let tq = CompositeChannelBuilder::two_qubit("tq", prob(0.01, pauli()));
//! ```
//!
//! ## Measurement Noise
//!
//! Bit-flip errors on measurement outcomes:
//!
//! ```
//! # use pecos_neo::noise::prelude::*;
//! // Symmetric (same error rate for 0->1 and 1->0)
//! let model = measurement_only(0.02, 0.02);
//!
//! // Asymmetric
//! let model = measurement_only(0.01, 0.03);  // p(0->1)=1%, p(1->0)=3%
//!
//! // Using primitives for outcome-dependent noise
//! let meas_noise = seq![
//!     on_zero(prob(0.01, flip_outcome())),  // 1% flip when measuring 0
//!     on_one(prob(0.03, flip_outcome())),   // 3% flip when measuring 1
//! ];
//! ```
//!
//! ## Leakage
//!
//! Model qubits leaving the computational basis:
//!
//! ```
//! # use pecos_neo::noise::prelude::*;
//! let model = with_leakage(
//!     0.001,  // p1: single-qubit error rate
//!     0.01,   // p2: two-qubit error rate
//!     0.1,    // 10% of errors cause leakage
//!     0.5,    // 50% seepage rate (return from leaked state)
//! );
//! ```
//!
//! ## Idle/Decoherence Noise
//!
//! T1 and T2 decay during idle time:
//!
//! ```
//! # use pecos_neo::noise::prelude::*;
//! // Using builder
//! let model = NoiseModelBuilder::new()
//!     .with_idle_noise(0.0001, 0.0005)  // T1 rate, T2 rate
//!     .build();
//!
//! // Using primitives for custom idle behavior
//! let t1 = CompositeChannelBuilder::idle("t1", prob_linear(0.0001, pauli()));
//! let t2 = CompositeChannelBuilder::idle("t2", prob_linear(0.0005, inject_z()));
//! ```
//!
//! ## Crosstalk
//!
//! Operations affecting neighboring qubits:
//!
//! ```
//! # use pecos_neo::noise::prelude::*;
//! // 1D chain crosstalk during measurement
//! let model = chain_measurement_crosstalk(0.01);
//!
//! // 2D grid crosstalk (5 columns)
//! let model = grid_measurement_crosstalk(5, 0.01);
//!
//! // Custom crosstalk with composite primitives
//! let crosstalk = CompositeCrosstalkChannel::new("custom", prob(0.01, inject_z()))
//!     .responds_to_measurement()
//!     .local(chain_neighbors);  // Only affect adjacent qubits
//! ```
//!
//! ## Correlated Errors
//!
//! Errors that spread between qubits:
//!
//! ```
//! # use pecos_neo::noise::prelude::*;
//! // Chain-correlated errors
//! let model = chain_correlated(0.01, 0.5);  // 50% correlation factor
//!
//! // Using CorrelatedNoiseChannel directly
//! let channel = CorrelatedNoiseChannel::new(0.01, 0.5);
//! ```
//!
//! # Topology Helpers
//!
//! For spatial noise models, use topology helpers:
//!
//! ```no_run
//! # use pecos_neo::noise::prelude::*;
//! # use pecos_neo::prelude::QubitId;
//! // Neighbor functions for crosstalk
//! let _neighbors = chain_neighbors;      // 1D: qubit i neighbors i-1, i+1
//! let _neighbors = grid_neighbors(5);    // 2D grid with 5 columns
//!
//! // Distance functions for correlation decay
//! let (a, b) = (QubitId(0), QubitId(3));
//! let _d = chain_distance(a, b);       // |i - j|
//! let _d = grid_distance(5)(a, b);     // Manhattan distance on grid
//!
//! // Decay functions for distance-weighted correlations
//! let decay = exponential_decay(0.5, 2.0);  // 0.5 * exp(-d/2)
//! let decay = gaussian_decay(1.0, 3.0);     // exp(-(d/3)^2)
//! let decay = power_law_decay(1.0, 2.0);    // 1/(1+d)^2
//! ```
//!
//! # Event Types
//!
//! Noise channels respond to these events:
//!
//! | Event | When | Common Use |
//! |-------|------|------------|
//! | `BeforeGate` | Before gate execution | Skip leaked qubits |
//! | `AfterGate` | After gate execution | Inject Pauli errors |
//! | `AfterPreparation` | After state prep | Preparation errors |
//! | `BeforeMeasurement` | Before measurement | Pre-measurement noise |
//! | `AfterMeasurement` | After measurement | Flip outcomes |
//! | `AfterReset` | After mid-circuit reset | Reset errors |
//! | `IdleTime` | During idle periods | T1/T2 decay |
//! | `BeforeCircuit` | Circuit start | Initial errors |
//! | `AfterCircuit` | Circuit end | Final errors |
//! | `BetweenLayers` | Between circuit layers | Layer idle noise |
//!
//! # Composite Primitives Reference
//!
//! ## Control Flow
//! - `prob(p, inner)` - Apply inner with probability p
//! - `when(cond, if_true, if_false)` - Conditional branching
//! - `seq![a, b, c]` - Sequential composition
//! - `sample![(w1, a), (w2, b)]` - Weighted random selection
//!
//! ## Conditions
//! - `leaked()` - Qubit is in leaked state
//! - `not_leaked()` - Qubit is not leaked
//! - `partner_leaked()` - Partner qubit is leaked (2Q gates)
//! - `outcome_is(val)` - Measurement outcome equals value
//!
//! ## Actions
//! - `pauli()` - Random Pauli (X, Y, or Z)
//! - `inject_x()`, `inject_y()`, `inject_z()` - Specific Pauli
//! - `leak()` - Cause leakage
//! - `seep()` - Return from leaked state
//! - `flip_outcome()` - Flip measurement result
//! - `nothing()` - No effect
//!
//! ## Convenience
//! - `skip_if_leaked()` - Skip gate if qubit is leaked
//! - `when_leaked(if_leaked, if_not)` - Branch on leakage
//! - `on_zero(action)` - Apply if outcome is 0
//! - `on_one(action)` - Apply if outcome is 1

// ============================================================================
// Unified Builder
// ============================================================================

pub use super::builder::NoiseModelBuilder;

// ============================================================================
// Composable Model
// ============================================================================

pub use super::composer::ComposableNoiseModel;

// ============================================================================
// Base Channels (Pre-built)
// ============================================================================

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

// ============================================================================
// Composite System (Composition)
// Re-export everything from the composite prelude
// ============================================================================

pub use super::composite::prelude::*;

// ============================================================================
// Core Traits and Types
// ============================================================================

pub use super::{GateInfo, IdleInfo, NoiseContext};
pub use super::{NoiseChannel, NoiseEvent, NoiseResponse};

// ============================================================================
// Topology (Spatial Noise Helpers)
// ============================================================================

pub use super::topology::{
    chain_distance, chain_neighbors, exponential_decay, gaussian_decay, grid_distance,
    grid_neighbors, power_law_decay,
};

// ============================================================================
// Convenience Patterns (Pre-built Configurations)
// ============================================================================

pub use super::patterns::{
    DeviceNoiseParams, chain_correlated, chain_measurement_crosstalk, dephasing_only,
    depolarizing_only, depolarizing_with_measurement, grid_measurement_crosstalk, measurement_only,
    realistic_device_noise, surface_code_noise, with_leakage,
};

// ============================================================================
// Validation
// ============================================================================

pub use super::validation::{
    ValidationError, ValidationResult, clamp_probability, is_probability_one, is_probability_zero,
    validate_probability, validate_rate, validate_weights,
};
