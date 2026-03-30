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

//! Pre-built noise model patterns for common use cases.
//!
//! This module provides ready-to-use noise configurations that cover common
//! quantum computing noise scenarios. These can be used directly or as starting
//! points for custom noise models.
//!
//! # Quick Reference
//!
//! | Pattern | Description |
//! |---------|-------------|
//! | [`depolarizing_only`] | Simple Pauli errors after gates |
//! | [`depolarizing_with_measurement`] | Gate + measurement errors |
//! | [`measurement_only`] | Just readout errors |
//! | [`dephasing_only`] | Z errors only (T2-like) |
//! | [`with_leakage`] | Leakage to non-computational states |
//! | [`chain_correlated`] | Spatially correlated errors (1D) |
//! | [`chain_measurement_crosstalk`] | Measurement affects neighbors (1D) |
//! | [`grid_measurement_crosstalk`] | Measurement affects neighbors (2D) |
//! | [`realistic_device_noise`] | Full device model with all parameters |
//! | [`surface_code_noise`] | Optimized for surface code simulations |
//!
//! # Examples
//!
//! ## Simple Noise
//!
//! ```
//! use pecos_neo::noise::prelude::*;
//!
//! // One-liner for basic depolarizing
//! let model = depolarizing_only(0.001, 0.01);
//!
//! // Add measurement errors
//! let model = depolarizing_with_measurement(0.001, 0.01, 0.02);
//! ```
//!
//! ## Asymmetric Measurement
//!
//! ```
//! use pecos_neo::noise::prelude::*;
//!
//! // Different error rates for 0->1 vs 1->0
//! let model = measurement_only(0.01, 0.05);  // 1% and 5%
//! ```
//!
//! ## Device Noise
//!
//! ```
//! use pecos_neo::noise::prelude::*;
//!
//! let model = realistic_device_noise(
//!     DeviceNoiseParams::new()
//!         .with_p1(0.001)           // 0.1% single-qubit error
//!         .with_p2(0.01)            // 1% two-qubit error
//!         .with_measurement_error(0.02)
//!         .with_prep_error(0.001)
//!         .with_t1(0.0001)          // T1 decay rate
//!         .with_t2(0.0005)          // T2 dephasing rate
//! );
//! ```
//!
//! ## Spatial Noise
//!
//! ```
//! use pecos_neo::noise::prelude::*;
//!
//! // Errors that spread between qubits
//! let model = chain_correlated(0.01, 0.5);  // 50% correlation
//!
//! // Measurement crosstalk on a grid
//! let model = grid_measurement_crosstalk(5, 0.01);  // 5 columns
//! ```

use super::CorrelatedNoiseChannel;
use super::builder::NoiseModelBuilder;
use super::composer::ComposableNoiseModel;
use super::composite::prelude::*;
use super::topology::{chain_neighbors, grid_neighbors};

// ============================================================================
// Simple Patterns
// ============================================================================

/// Create a noise model with only depolarizing noise.
///
/// This is the simplest useful noise model - just Pauli errors after gates.
///
/// # Arguments
///
/// * `p1` - Single-qubit gate error probability
/// * `p2` - Two-qubit gate error probability
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::prelude::*;
/// let model = depolarizing_only(0.001, 0.01);
/// ```
#[must_use]
pub fn depolarizing_only(p1: f64, p2: f64) -> ComposableNoiseModel {
    NoiseModelBuilder::new().with_depolarizing(p1, p2).build()
}

/// Create a noise model with depolarizing and measurement noise.
///
/// # Arguments
///
/// * `p1` - Single-qubit gate error probability
/// * `p2` - Two-qubit gate error probability
/// * `p_meas` - Symmetric measurement error probability
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::prelude::*;
/// let model = depolarizing_with_measurement(0.001, 0.01, 0.02);
/// ```
#[must_use]
pub fn depolarizing_with_measurement(p1: f64, p2: f64, p_meas: f64) -> ComposableNoiseModel {
    NoiseModelBuilder::new()
        .with_depolarizing(p1, p2)
        .with_measurement_error(p_meas)
        .build()
}

/// Create a noise model with only measurement noise (no gate errors).
///
/// Useful for studying measurement error effects in isolation.
///
/// # Arguments
///
/// * `p01` - Probability of measuring 1 when state is 0
/// * `p10` - Probability of measuring 0 when state is 1
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::prelude::*;
/// // Symmetric measurement error
/// let model = measurement_only(0.02, 0.02);
///
/// // Asymmetric measurement error
/// let model = measurement_only(0.01, 0.03);
/// ```
#[must_use]
pub fn measurement_only(p01: f64, p10: f64) -> ComposableNoiseModel {
    NoiseModelBuilder::new()
        .with_measurement_error_asymmetric(p01, p10)
        .build()
}

/// Create a noise model with pure dephasing (Z errors only).
///
/// Models T2 dephasing without energy relaxation.
///
/// # Arguments
///
/// * `p1` - Single-qubit dephasing probability
/// * `p2` - Two-qubit dephasing probability
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::prelude::*;
/// let model = dephasing_only(0.001, 0.01);
/// ```
#[must_use]
pub fn dephasing_only(p1: f64, p2: f64) -> ComposableNoiseModel {
    let sq_noise = prob(p1, inject_z());
    // For two-qubit dephasing: ZI, IZ, or ZZ
    let tq_noise = prob(p2, inject_z());

    let sq_channel = CompositeChannelBuilder::single_qubit("sq_dephasing", sq_noise);
    let tq_channel = CompositeChannelBuilder::two_qubit("tq_dephasing", tq_noise);

    ComposableNoiseModel::new()
        .add_channel(sq_channel)
        .add_channel(tq_channel)
}

// ============================================================================
// Leakage Patterns
// ============================================================================

/// Create a noise model with leakage and seepage.
///
/// Models leakage to excited states and seepage back to computational basis.
///
/// # Arguments
///
/// * `p1` - Single-qubit error probability
/// * `p2` - Two-qubit error probability
/// * `emission_ratio` - Fraction of errors that cause leakage (vs Pauli)
/// * `seepage_rate` - Rate of seepage back from leaked state
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::prelude::*;
/// let model = with_leakage(0.001, 0.01, 0.1, 0.5);
/// ```
#[must_use]
pub fn with_leakage(
    p1: f64,
    p2: f64,
    emission_ratio: f64,
    seepage_rate: f64,
) -> ComposableNoiseModel {
    let pauli_ratio = 1.0 - emission_ratio;

    // Single-qubit noise with leakage
    let sq_noise = seq![
        skip_if_leaked(),
        prob(
            p1,
            when_leaked(
                prob(seepage_rate, seep()),
                sample![(emission_ratio, leak()), (pauli_ratio, pauli())],
            ),
        ),
    ];

    // Two-qubit noise with leakage
    // Note: For two-qubit gates, we use skip_if_leaked which skips if the current qubit is leaked
    let tq_noise = seq![
        skip_if_leaked(),
        prob(
            p2,
            when_leaked(
                prob(seepage_rate, seep()),
                sample![(emission_ratio, leak()), (pauli_ratio, pauli())],
            ),
        ),
    ];

    let sq_channel = CompositeChannelBuilder::single_qubit("sq_leakage", sq_noise);
    let tq_channel = CompositeChannelBuilder::two_qubit("tq_leakage", tq_noise);
    let before_channel = CompositeChannelBuilder::before_gate("skip_leaked", skip_if_leaked());

    ComposableNoiseModel::new()
        .add_channel(before_channel)
        .add_channel(sq_channel)
        .add_channel(tq_channel)
}

// ============================================================================
// Correlated Noise Patterns
// ============================================================================

/// Create a noise model with spatially correlated errors on a 1D chain.
///
/// Errors on one qubit increase the probability of errors on adjacent qubits.
///
/// # Arguments
///
/// * `base_probability` - Base error probability per qubit
/// * `correlation_factor` - How much errors correlate (0.0 = independent, 1.0 = fully correlated)
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::prelude::*;
/// let model = chain_correlated(0.01, 0.5);
/// ```
#[must_use]
pub fn chain_correlated(base_probability: f64, correlation_factor: f64) -> ComposableNoiseModel {
    let channel = CorrelatedNoiseChannel::new(base_probability, correlation_factor);
    ComposableNoiseModel::new().add_channel(channel)
}

/// Create a noise model with measurement crosstalk on a 1D chain.
///
/// Measuring a qubit can flip its neighbors.
///
/// # Arguments
///
/// * `crosstalk_probability` - Probability of flipping a neighbor during measurement
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::prelude::*;
/// let model = chain_measurement_crosstalk(0.01);
/// ```
#[must_use]
pub fn chain_measurement_crosstalk(crosstalk_probability: f64) -> ComposableNoiseModel {
    let crosstalk =
        CompositeCrosstalkChannel::new("chain_crosstalk", prob(crosstalk_probability, pauli()))
            .responds_to_measurement()
            .local(chain_neighbors);

    ComposableNoiseModel::new().add_channel(crosstalk)
}

/// Create a noise model with measurement crosstalk on a 2D grid.
///
/// # Arguments
///
/// * `cols` - Number of columns in the grid
/// * `crosstalk_probability` - Probability of flipping a neighbor during measurement
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::prelude::*;
/// // 5x4 grid with crosstalk
/// let model = grid_measurement_crosstalk(5, 0.01);
/// ```
#[must_use]
pub fn grid_measurement_crosstalk(cols: usize, crosstalk_probability: f64) -> ComposableNoiseModel {
    let crosstalk =
        CompositeCrosstalkChannel::new("grid_crosstalk", prob(crosstalk_probability, pauli()))
            .responds_to_measurement()
            .local(grid_neighbors(cols));

    ComposableNoiseModel::new().add_channel(crosstalk)
}

// ============================================================================
// Realistic Device Patterns
// ============================================================================

/// Parameters for realistic device noise.
#[derive(Debug, Clone)]
pub struct DeviceNoiseParams {
    /// Single-qubit gate error probability.
    pub p1: f64,
    /// Two-qubit gate error probability.
    pub p2: f64,
    /// Measurement error probability (0 -> 1).
    pub p_meas_01: f64,
    /// Measurement error probability (1 -> 0).
    pub p_meas_10: f64,
    /// Preparation error probability.
    pub p_prep: f64,
    /// Leakage probability (fraction of errors that leak).
    pub leakage_rate: f64,
    /// Seepage rate (probability of returning from leaked state).
    pub seepage_rate: f64,
    /// T1 idle decay rate (per time unit).
    pub t1_rate: f64,
    /// T2 idle dephasing rate (per time unit).
    pub t2_rate: f64,
}

impl Default for DeviceNoiseParams {
    fn default() -> Self {
        Self {
            p1: 0.001,
            p2: 0.01,
            p_meas_01: 0.02,
            p_meas_10: 0.02,
            p_prep: 0.001,
            leakage_rate: 0.0,
            seepage_rate: 0.0,
            t1_rate: 0.0,
            t2_rate: 0.0,
        }
    }
}

impl DeviceNoiseParams {
    /// Create new device noise parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set single-qubit gate error probability.
    #[must_use]
    pub fn with_p1(mut self, p1: f64) -> Self {
        self.p1 = p1;
        self
    }

    /// Set two-qubit gate error probability.
    #[must_use]
    pub fn with_p2(mut self, p2: f64) -> Self {
        self.p2 = p2;
        self
    }

    /// Set symmetric measurement error probability.
    #[must_use]
    pub fn with_measurement_error(mut self, p: f64) -> Self {
        self.p_meas_01 = p;
        self.p_meas_10 = p;
        self
    }

    /// Set asymmetric measurement error probabilities.
    #[must_use]
    pub fn with_asymmetric_measurement(mut self, p01: f64, p10: f64) -> Self {
        self.p_meas_01 = p01;
        self.p_meas_10 = p10;
        self
    }

    /// Set preparation error probability.
    #[must_use]
    pub fn with_prep_error(mut self, p: f64) -> Self {
        self.p_prep = p;
        self
    }

    /// Set leakage rate (fraction of errors that cause leakage).
    #[must_use]
    pub fn with_leakage(mut self, rate: f64) -> Self {
        self.leakage_rate = rate;
        self
    }

    /// Set seepage rate (probability of returning from leaked state).
    #[must_use]
    pub fn with_seepage(mut self, rate: f64) -> Self {
        self.seepage_rate = rate;
        self
    }

    /// Set T1 relaxation rate (per time unit).
    #[must_use]
    pub fn with_t1(mut self, rate: f64) -> Self {
        self.t1_rate = rate;
        self
    }

    /// Set T2 dephasing rate (per time unit).
    #[must_use]
    pub fn with_t2(mut self, rate: f64) -> Self {
        self.t2_rate = rate;
        self
    }
}

/// Create a realistic device noise model from parameters.
///
/// This builds a comprehensive noise model including:
/// - Gate depolarizing noise (with optional leakage)
/// - Measurement errors
/// - Preparation errors
/// - Idle T1/T2 decay (if rates are non-zero)
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::prelude::*;
/// let model = realistic_device_noise(
///     DeviceNoiseParams::new()
///         .with_p1(0.001)
///         .with_p2(0.01)
///         .with_measurement_error(0.02)
///         .with_t1(0.0001)
///         .with_t2(0.0005)
/// );
/// ```
#[must_use]
pub fn realistic_device_noise(params: DeviceNoiseParams) -> ComposableNoiseModel {
    let mut builder = NoiseModelBuilder::new();

    // Gate noise
    if params.leakage_rate > 0.0 {
        let pauli_ratio = 1.0 - params.leakage_rate;

        // Single-qubit noise with leakage
        let sq_noise = seq![
            skip_if_leaked(),
            prob(
                params.p1,
                when_leaked(
                    prob(params.seepage_rate, seep()),
                    sample![(params.leakage_rate, leak()), (pauli_ratio, pauli()),],
                ),
            ),
        ];

        builder = builder.with_single_qubit_noise(sq_noise);
    } else {
        builder = builder.with_depolarizing(params.p1, params.p2);
    }

    // Measurement noise
    if params.p_meas_01 > 0.0 || params.p_meas_10 > 0.0 {
        builder = builder.with_measurement_error_asymmetric(params.p_meas_01, params.p_meas_10);
    }

    // Preparation noise
    if params.p_prep > 0.0 {
        builder = builder.with_preparation_error(params.p_prep);
    }

    let mut model = builder.build();

    // Add idle noise if rates are set
    if params.t1_rate > 0.0 {
        // T1 decay: energy relaxation causes bit flips (amplitude damping approximated as Pauli)
        let t1_channel =
            CompositeChannelBuilder::idle("t1_decay", prob_linear(params.t1_rate, pauli()));
        model = model.add_channel(t1_channel);
    }

    if params.t2_rate > 0.0 {
        // T2 dephasing: pure dephasing (Z errors)
        let t2_channel =
            CompositeChannelBuilder::idle("t2_dephasing", prob_linear(params.t2_rate, inject_z()));
        model = model.add_channel(t2_channel);
    }

    model
}

// ============================================================================
// Surface Code Patterns
// ============================================================================

/// Create a noise model optimized for surface code simulations.
///
/// This model includes:
/// - Depolarizing noise scaled for surface code error rates
/// - Measurement errors (critical for surface code decoding)
/// - Optional crosstalk between data and ancilla qubits
///
/// # Arguments
///
/// * `physical_error_rate` - Base physical error rate (p)
/// * `with_crosstalk` - Whether to include measurement crosstalk
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::prelude::*;
/// let model = surface_code_noise(0.001, false);
/// ```
#[must_use]
pub fn surface_code_noise(physical_error_rate: f64, with_crosstalk: bool) -> ComposableNoiseModel {
    // For surface codes, 2Q errors are typically ~10x worse than 1Q
    let p1 = physical_error_rate;
    let p2 = physical_error_rate * 10.0;
    let p_meas = physical_error_rate * 2.0; // Measurement typically worse

    let mut model = NoiseModelBuilder::new()
        .with_depolarizing(p1, p2)
        .with_measurement_error(p_meas)
        .build();

    if with_crosstalk {
        // Add global measurement crosstalk for surface code
        let crosstalk = CompositeCrosstalkChannel::new(
            "meas_crosstalk",
            prob(physical_error_rate * 0.1, inject_z()),
        )
        .responds_to_measurement()
        .global();

        model = model.add_channel(crosstalk);
    }

    model
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::command::CommandBuilder;
    use crate::runner::CircuitRunner;
    use pecos_simulators::SparseStab;

    #[test]
    fn test_depolarizing_only() {
        let model = depolarizing_only(0.5, 0.5);
        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(model)
            .with_seed(42);

        // Just verify it runs
        let _ = runner.apply_circuit(&mut state, &commands).unwrap();
    }

    #[test]
    fn test_depolarizing_with_measurement() {
        let commands = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

        let mut state = SparseStab::new(1);
        let mut errors = 0;
        for seed in 0..100 {
            // Recreate model for each iteration since ComposableNoiseModel doesn't Clone
            let model = depolarizing_with_measurement(0.0, 0.0, 0.5);
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model)
                .with_seed(seed);
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            if outcomes
                .get(pecos_core::QubitId(0))
                .is_some_and(|o| o.outcome)
            {
                errors += 1;
            }
        }

        // Should be roughly 50% errors
        assert!(
            errors > 30 && errors < 70,
            "Expected ~50 errors, got {errors}"
        );
    }

    #[test]
    fn test_dephasing_only() {
        let model = dephasing_only(0.5, 0.5);
        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(model)
            .with_seed(42);

        let _ = runner.apply_circuit(&mut state, &commands).unwrap();
    }

    #[test]
    fn test_with_leakage() {
        let model = with_leakage(0.5, 0.5, 0.5, 0.5);
        let commands = CommandBuilder::new()
            .pz(&[0])
            .h(&[0])
            .h(&[0])
            .mz(&[0])
            .build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(model)
            .with_seed(42);

        let _ = runner.apply_circuit(&mut state, &commands).unwrap();
    }

    #[test]
    fn test_chain_correlated() {
        let model = chain_correlated(0.1, 0.5);
        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .h(&[0])
            .h(&[1])
            .mz(&[0])
            .mz(&[1])
            .build();

        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(model)
            .with_seed(42);

        let _ = runner.apply_circuit(&mut state, &commands).unwrap();
    }

    #[test]
    fn test_realistic_device_noise() {
        let params = DeviceNoiseParams::new()
            .with_p1(0.001)
            .with_p2(0.01)
            .with_measurement_error(0.02);

        let model = realistic_device_noise(params);
        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(model)
            .with_seed(42);

        let _ = runner.apply_circuit(&mut state, &commands).unwrap();
    }

    #[test]
    fn test_surface_code_noise() {
        let model = surface_code_noise(0.001, true);
        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .cx(&[(0, 1)])
            .mz(&[0])
            .mz(&[1])
            .build();

        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(model)
            .with_seed(42);

        let _ = runner.apply_circuit(&mut state, &commands).unwrap();
    }

    #[test]
    fn test_device_params_builder() {
        let params = DeviceNoiseParams::new()
            .with_p1(0.001)
            .with_p2(0.01)
            .with_asymmetric_measurement(0.01, 0.03)
            .with_prep_error(0.005)
            .with_leakage(0.1)
            .with_seepage(0.5)
            .with_t1(0.0001)
            .with_t2(0.0005);

        assert_eq!(params.p1, 0.001);
        assert_eq!(params.p2, 0.01);
        assert_eq!(params.p_meas_01, 0.01);
        assert_eq!(params.p_meas_10, 0.03);
        assert_eq!(params.p_prep, 0.005);
        assert_eq!(params.leakage_rate, 0.1);
        assert_eq!(params.seepage_rate, 0.5);
        assert_eq!(params.t1_rate, 0.0001);
        assert_eq!(params.t2_rate, 0.0005);
    }
}
