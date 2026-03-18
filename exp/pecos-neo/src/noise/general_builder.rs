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

//! Builder for a `GeneralNoiseModel`-equivalent noise configuration.
//!
//! This is a convenience wrapper that produces a [`ComposableNoiseModel`] configured
//! with the same parameters as the original `GeneralNoiseModel` from `pecos-engines`.
//!
//! # Example
//!
//! ```
//! use pecos_neo::noise::GeneralNoiseModelBuilder;
//!
//! let noise = GeneralNoiseModelBuilder::new()
//!     .with_p1(0.001)
//!     .with_p2(0.01)
//!     .with_p_meas(0.02, 0.03)
//!     .with_p_prep(0.005)
//!     .build();
//! ```
//!
//! # Composability
//!
//! This builder is just one way to configure a noise model. You can also:
//! - Compose channels directly with [`ComposableNoiseModel`]
//! - Create your own builders for different noise patterns
//! - Mix and match: start with this builder and add custom channels

use super::crosstalk::CrosstalkChannel;
use super::idle::IdleChannel;
use super::leakage::LeakageChannel;
use super::measurement::MeasurementChannel;
use super::plugins::CorePlugin;
use super::preparation::PreparationChannel;
use super::single_qubit::SingleQubitChannel;
use super::two_qubit::{AngleScaling, TwoQubitChannel};
use super::{
    ComposableNoiseModel, CrosstalkTransitions, PauliWeights, SingleQubitEmissionWeights,
    TwoQubitEmissionWeights, TwoQubitPauliWeights,
};
use crate::command::GateType;
use pecos_core::TimeScale;

/// Builder for creating a noise model equivalent to `GeneralNoiseModel`.
///
/// This provides a familiar API for users coming from `GeneralNoiseModel` while
/// using the composable channel architecture underneath.
///
/// # Mixing Channel Types
///
/// You can mix traditional channels with composite channels using [`with_channel`]:
///
/// ```no_run
/// use pecos_neo::noise::GeneralNoiseModelBuilder;
/// use pecos_neo::noise::composite::prelude::*;
///
/// let model = GeneralNoiseModelBuilder::new()
///     .with_p1(0.001)                    // Traditional 1Q channel
///     .with_p_meas(0.02, 0.03)           // Traditional measurement channel
///     .with_channel(                      // Custom composite channel for 2Q
///         CompositeChannelBuilder::two_qubit("custom_2q", seq![
///             skip_if_leaked(),
///             prob(0.01, pauli()),
///         ])
///     )
///     .build();
/// ```
///
/// [`with_channel`]: Self::with_channel
pub struct GeneralNoiseModelBuilder {
    // Preparation
    p_prep: f64,
    p_prep_leak_ratio: f64,
    p_prep_crosstalk: f64,

    // Single-qubit gates
    p1: f64,
    p1_emission_ratio: f64,
    p1_emission_weights: SingleQubitEmissionWeights,
    p1_pauli_weights: PauliWeights,
    p1_seepage_prob: f64,

    // Two-qubit gates
    p2: f64,
    p2_angle_scaling: AngleScaling,
    p2_emission_ratio: f64,
    p2_emission_weights: TwoQubitEmissionWeights,
    p2_pauli_weights: TwoQubitPauliWeights,
    p2_seepage_prob: f64,
    p2_idle: f64,

    // Measurement
    p_meas_0: f64,
    p_meas_1: f64,
    p_meas_crosstalk_global: f64,
    p_meas_crosstalk_local: f64,
    p_meas_crosstalk_transitions: Option<CrosstalkTransitions>,

    // Idle noise
    p_idle_linear_rate: f64,
    p_idle_linear_weights: PauliWeights,
    p_idle_quadratic_rate: f64,
    p_idle_coherent: bool,
    p_idle_coherent_to_incoherent_factor: f64,

    // Leakage
    leakage_scale: f64,

    // Noiseless gates
    noiseless_gates: Vec<GateType>,

    // Time scale for physical time interpretation
    time_scale: Option<TimeScale>,

    // Custom channels (composite or traditional)
    custom_channels: Vec<Box<dyn super::NoiseChannel>>,
}

impl Default for GeneralNoiseModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneralNoiseModelBuilder {
    /// Create a new builder with all parameters set to zero/default.
    #[must_use]
    pub fn new() -> Self {
        Self {
            // Preparation
            p_prep: 0.0,
            p_prep_leak_ratio: 0.0,
            p_prep_crosstalk: 0.0,

            // Single-qubit
            p1: 0.0,
            p1_emission_ratio: 0.0,
            p1_emission_weights: SingleQubitEmissionWeights::uniform(),
            p1_pauli_weights: PauliWeights::uniform(),
            p1_seepage_prob: 0.0,

            // Two-qubit
            p2: 0.0,
            p2_angle_scaling: AngleScaling::constant(),
            p2_emission_ratio: 0.0,
            p2_emission_weights: TwoQubitEmissionWeights::uniform_pauli(),
            p2_pauli_weights: TwoQubitPauliWeights::uniform(),
            p2_seepage_prob: 0.0,
            p2_idle: 0.0,

            // Measurement
            p_meas_0: 0.0,
            p_meas_1: 0.0,
            p_meas_crosstalk_global: 0.0,
            p_meas_crosstalk_local: 0.0,
            p_meas_crosstalk_transitions: None,

            // Idle
            p_idle_linear_rate: 0.0,
            p_idle_linear_weights: PauliWeights::custom(0.0, 0.0, 1.0), // Z-only
            p_idle_quadratic_rate: 0.0,
            p_idle_coherent: false,
            p_idle_coherent_to_incoherent_factor: 1.0,

            // Leakage
            leakage_scale: 1.0,

            // Noiseless
            noiseless_gates: Vec::new(),

            // Time scale
            time_scale: None,

            // Custom channels
            custom_channels: Vec::new(),
        }
    }

    // ========================================================================
    // Custom channels
    // ========================================================================

    /// Add a custom noise channel (composite or traditional).
    ///
    /// This allows mixing different channel types in a single noise model.
    /// Channels are applied in the order they are added.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::noise::GeneralNoiseModelBuilder;
    /// use pecos_neo::noise::composite::prelude::*;
    ///
    /// let model = GeneralNoiseModelBuilder::new()
    ///     .with_p1(0.001)  // Traditional single-qubit noise
    ///     .with_channel(   // Custom composite channel
    ///         CompositeChannelBuilder::two_qubit("leaky_2q", seq![
    ///             skip_if_leaked(),
    ///             prob(0.01, when_leaked(seep(), pauli())),
    ///         ])
    ///     )
    ///     .build();
    /// ```
    #[must_use]
    pub fn with_channel(mut self, channel: impl super::NoiseChannel + 'static) -> Self {
        self.custom_channels.push(Box::new(channel));
        self
    }

    // ========================================================================
    // Preparation parameters
    // ========================================================================

    /// Set the preparation error probability.
    #[must_use]
    pub fn with_p_prep(mut self, p: f64) -> Self {
        self.p_prep = p;
        self
    }

    /// Set the fraction of preparation errors that cause leakage.
    #[must_use]
    pub fn with_p_prep_leak_ratio(mut self, ratio: f64) -> Self {
        self.p_prep_leak_ratio = ratio;
        self
    }

    /// Set the preparation crosstalk probability.
    #[must_use]
    pub fn with_p_prep_crosstalk(mut self, p: f64) -> Self {
        self.p_prep_crosstalk = p;
        self
    }

    // ========================================================================
    // Single-qubit gate parameters
    // ========================================================================

    /// Set the single-qubit gate error probability.
    #[must_use]
    pub fn with_p1(mut self, p: f64) -> Self {
        self.p1 = p;
        self
    }

    /// Set the fraction of single-qubit errors that are emission errors.
    #[must_use]
    pub fn with_p1_emission_ratio(mut self, ratio: f64) -> Self {
        self.p1_emission_ratio = ratio;
        self
    }

    /// Set the emission error distribution for single-qubit gates.
    #[must_use]
    pub fn with_p1_emission_weights(mut self, weights: SingleQubitEmissionWeights) -> Self {
        self.p1_emission_weights = weights;
        self
    }

    /// Set the Pauli error distribution for single-qubit gates.
    #[must_use]
    pub fn with_p1_pauli_weights(mut self, weights: PauliWeights) -> Self {
        self.p1_pauli_weights = weights;
        self
    }

    /// Set the seepage probability for single-qubit gates.
    #[must_use]
    pub fn with_p1_seepage(mut self, p: f64) -> Self {
        self.p1_seepage_prob = p;
        self
    }

    // ========================================================================
    // Two-qubit gate parameters
    // ========================================================================

    /// Set the two-qubit gate error probability.
    #[must_use]
    pub fn with_p2(mut self, p: f64) -> Self {
        self.p2 = p;
        self
    }

    /// Set angle-dependent scaling for two-qubit gates.
    #[must_use]
    pub fn with_p2_angle_scaling(mut self, scaling: AngleScaling) -> Self {
        self.p2_angle_scaling = scaling;
        self
    }

    /// Set the fraction of two-qubit errors that are emission errors.
    #[must_use]
    pub fn with_p2_emission_ratio(mut self, ratio: f64) -> Self {
        self.p2_emission_ratio = ratio;
        self
    }

    /// Set the emission error distribution for two-qubit gates.
    #[must_use]
    pub fn with_p2_emission_weights(mut self, weights: TwoQubitEmissionWeights) -> Self {
        self.p2_emission_weights = weights;
        self
    }

    /// Set the Pauli error distribution for two-qubit gates.
    #[must_use]
    pub fn with_p2_pauli_weights(mut self, weights: TwoQubitPauliWeights) -> Self {
        self.p2_pauli_weights = weights;
        self
    }

    /// Set the seepage probability for two-qubit gates.
    #[must_use]
    pub fn with_p2_seepage(mut self, p: f64) -> Self {
        self.p2_seepage_prob = p;
        self
    }

    /// Set idle noise rate applied after two-qubit gates.
    #[must_use]
    pub fn with_p2_idle(mut self, rate: f64) -> Self {
        self.p2_idle = rate;
        self
    }

    // ========================================================================
    // Measurement parameters
    // ========================================================================

    /// Set asymmetric measurement error probabilities.
    #[must_use]
    pub fn with_p_meas(mut self, p_0_to_1: f64, p_1_to_0: f64) -> Self {
        self.p_meas_0 = p_0_to_1;
        self.p_meas_1 = p_1_to_0;
        self
    }

    /// Set symmetric measurement error probability.
    #[must_use]
    pub fn with_p_meas_symmetric(mut self, p: f64) -> Self {
        self.p_meas_0 = p;
        self.p_meas_1 = p;
        self
    }

    /// Set measurement crosstalk probabilities (global and local).
    #[must_use]
    pub fn with_p_meas_crosstalk(mut self, global: f64, local: f64) -> Self {
        self.p_meas_crosstalk_global = global;
        self.p_meas_crosstalk_local = local;
        self
    }

    /// Set measurement crosstalk transition model.
    #[must_use]
    pub fn with_p_meas_crosstalk_transitions(mut self, transitions: CrosstalkTransitions) -> Self {
        self.p_meas_crosstalk_transitions = Some(transitions);
        self
    }

    // ========================================================================
    // Idle noise parameters
    // ========================================================================

    /// Set the linear idle noise rate (per time unit).
    ///
    /// The rate interpretation depends on your `TimeScale` configuration.
    #[must_use]
    pub fn with_p_idle_linear(mut self, rate: f64) -> Self {
        self.p_idle_linear_rate = rate;
        self
    }

    /// Set the Pauli distribution for linear idle noise.
    #[must_use]
    pub fn with_p_idle_linear_weights(mut self, weights: PauliWeights) -> Self {
        self.p_idle_linear_weights = weights;
        self
    }

    /// Set the quadratic idle noise rate (per time unit).
    ///
    /// The rate interpretation depends on your `TimeScale` configuration.
    #[must_use]
    pub fn with_p_idle_quadratic(mut self, rate: f64) -> Self {
        self.p_idle_quadratic_rate = rate;
        self
    }

    /// Set whether to use coherent dephasing for quadratic idle noise.
    #[must_use]
    pub fn with_p_idle_coherent(mut self, coherent: bool) -> Self {
        self.p_idle_coherent = coherent;
        self
    }

    /// Set the coherent-to-incoherent conversion factor.
    #[must_use]
    pub fn with_p_idle_coherent_to_incoherent_factor(mut self, factor: f64) -> Self {
        self.p_idle_coherent_to_incoherent_factor = factor;
        self
    }

    // ========================================================================
    // Leakage parameters
    // ========================================================================

    /// Set the leakage scale (0.0 = no leakage, 1.0 = full leakage).
    #[must_use]
    pub fn with_leakage_scale(mut self, scale: f64) -> Self {
        self.leakage_scale = scale;
        self
    }

    // ========================================================================
    // Noiseless gates
    // ========================================================================

    /// Mark a gate type as noiseless.
    #[must_use]
    pub fn with_noiseless_gate(mut self, gate_type: GateType) -> Self {
        self.noiseless_gates.push(gate_type);
        self
    }

    /// Mark multiple gate types as noiseless.
    #[must_use]
    pub fn with_noiseless_gates(mut self, gate_types: &[GateType]) -> Self {
        self.noiseless_gates.extend_from_slice(gate_types);
        self
    }

    // ========================================================================
    // Time scale
    // ========================================================================

    /// Set the time scale for interpreting physical time parameters.
    ///
    /// When set, convenience methods like `with_idle_t1_t2()` become available.
    ///
    /// # Example
    /// ```
    /// use pecos_neo::noise::GeneralNoiseModelBuilder;
    /// use pecos_core::TimeScale;
    ///
    /// let noise = GeneralNoiseModelBuilder::new()
    ///     .with_time_scale(TimeScale::NANOSECONDS)
    ///     .with_idle_t1_t2(50e-6, 30e-6)  // T1=50us, T2=30us
    ///     .build();
    /// ```
    #[must_use]
    pub fn with_time_scale(mut self, scale: TimeScale) -> Self {
        self.time_scale = Some(scale);
        self
    }

    /// Set T1/T2 relaxation times in physical units (seconds).
    ///
    /// Requires `with_time_scale()` to be called first.
    ///
    /// # Panics
    /// Panics if `with_time_scale()` has not been called.
    #[must_use]
    pub fn with_idle_t1_t2(mut self, t1_seconds: f64, t2_seconds: f64) -> Self {
        let scale = self
            .time_scale
            .expect("with_time_scale() must be called before with_idle_t1_t2()");

        // Convert physical times to time units
        let t1_units = scale.from_seconds(t1_seconds).as_f64();
        let t2_units = scale.from_seconds(t2_seconds).as_f64();

        // Set rates: linear_rate = 1/T1, quadratic_rate = 1/T2^2
        self.p_idle_linear_rate = 1.0 / t1_units.max(1.0);
        self.p_idle_quadratic_rate = 1.0 / (t2_units * t2_units).max(1.0);
        self
    }

    // ========================================================================
    // Build
    // ========================================================================

    /// Check if any configured parameters can cause leakage.
    fn has_leakage_potential(&self) -> bool {
        self.p_prep_leak_ratio > 0.0
            || self.p1_emission_ratio > 0.0
            || self.p2_emission_ratio > 0.0
            || self
                .p_meas_crosstalk_transitions
                .as_ref()
                .is_some_and(|t| t.from_0_leak > 0.0 || t.from_1_leak > 0.0)
    }

    /// Build the configured noise model.
    ///
    /// Returns a [`ComposableNoiseModel`] with all the configured channels.
    #[must_use]
    pub fn build(self) -> ComposableNoiseModel {
        let mut model = ComposableNoiseModel::new().add_plugin(CorePlugin);

        // Set time scale if configured
        if let Some(scale) = self.time_scale {
            model = model.with_time_scale(scale);
        }

        // Add noiseless gates
        for gate_type in &self.noiseless_gates {
            model = model.with_noiseless_gate(*gate_type);
        }

        // Leakage channel (handles leaked qubit effects) - only add if leakage is possible
        if self.leakage_scale > 0.0 && self.has_leakage_potential() {
            model = model.add_channel(LeakageChannel::new().with_scale(self.leakage_scale));
        }

        // Preparation channel
        if self.p_prep > 0.0 {
            model = model.add_channel(
                PreparationChannel::new(self.p_prep).with_leakage(self.p_prep_leak_ratio),
            );
        }

        // Single-qubit channel
        if self.p1 > 0.0 {
            let channel = SingleQubitChannel::new(
                self.p1,
                self.p1_pauli_weights,
                self.p1_emission_ratio,
                self.p1_emission_weights,
                self.p1_seepage_prob,
            );
            model = model.add_channel(channel);
        }

        // Two-qubit channel
        if self.p2 > 0.0 {
            let channel = TwoQubitChannel::new(
                self.p2,
                self.p2_angle_scaling,
                self.p2_pauli_weights,
                self.p2_emission_ratio,
                self.p2_emission_weights,
                self.p2_seepage_prob,
                self.p2_idle,
            );
            model = model.add_channel(channel);
        }

        // Measurement channel
        if self.p_meas_0 > 0.0 || self.p_meas_1 > 0.0 {
            model = model.add_channel(MeasurementChannel::asymmetric(self.p_meas_0, self.p_meas_1));
        }

        // Crosstalk channel (handles both prep and measurement crosstalk)
        if self.p_meas_crosstalk_global > 0.0
            || self.p_meas_crosstalk_local > 0.0
            || self.p_prep_crosstalk > 0.0
        {
            let mut crosstalk = CrosstalkChannel::new()
                .with_global_rate(self.p_meas_crosstalk_global.max(self.p_prep_crosstalk))
                .with_local_rate(self.p_meas_crosstalk_local);

            if let Some(transitions) = self.p_meas_crosstalk_transitions {
                crosstalk = crosstalk.with_transitions(transitions);
            }

            model = model.add_channel(crosstalk);
        }

        // Idle channel
        if self.p_idle_linear_rate > 0.0 || self.p_idle_quadratic_rate > 0.0 {
            let channel = IdleChannel {
                linear_rate: self.p_idle_linear_rate,
                linear_weights: self.p_idle_linear_weights,
                quadratic_rate: self.p_idle_quadratic_rate,
                coherent_dephasing: self.p_idle_coherent,
                coherent_to_incoherent_factor: self.p_idle_coherent_to_incoherent_factor,
            };
            model = model.add_channel(channel);
        }

        // Custom channels (composite or traditional)
        for channel in self.custom_channels {
            model = model.add_boxed_channel(channel);
        }

        model
    }
}

/// Create a general noise model builder.
///
/// This is a convenience entry point equivalent to [`GeneralNoiseModelBuilder::new()`],
/// providing API consistency with other free functions like [`sparse_stab()`](crate::tool::sparse_stab)
/// and [`state_vector()`](crate::tool::state_vector).
///
/// # Example
///
/// ```
/// use pecos_neo::noise::general_noise;
///
/// let noise = general_noise()
///     .with_p1(0.001)
///     .with_p2(0.01)
///     .with_p_meas(0.02, 0.03)
///     .with_p_prep(0.005)
///     .build();
/// ```
#[must_use]
pub fn general_noise() -> GeneralNoiseModelBuilder {
    GeneralNoiseModelBuilder::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_builder() {
        let model = GeneralNoiseModelBuilder::new().build();
        // Should have CorePlugin's handlers but no noise channels
        assert_eq!(model.event_handler_count(), 2); // Prep + Meas handlers
        assert_eq!(model.channel_count(), 0);
    }

    #[test]
    fn test_general_noise_equivalence() {
        // general_noise() should produce identical results to GeneralNoiseModelBuilder::new()
        use crate::command::CommandBuilder;
        use crate::runner::CircuitRunner;
        use pecos_core::QubitId;
        use pecos_qsim::SparseStab;

        let commands = CommandBuilder::new().pz(0).z(0).mz(0).build();

        let noise_a = general_noise()
            .with_p1(0.3)
            .with_p_meas_symmetric(0.1)
            .build();

        let noise_b = GeneralNoiseModelBuilder::new()
            .with_p1(0.3)
            .with_p_meas_symmetric(0.1)
            .build();

        let mut state_a = SparseStab::new(1);
        let mut runner_a = CircuitRunner::<SparseStab>::new()
            .with_noise(noise_a)
            .with_seed(42);

        let mut state_b = SparseStab::new(1);
        let mut runner_b = CircuitRunner::<SparseStab>::new()
            .with_noise(noise_b)
            .with_seed(42);

        for _ in 0..50 {
            state_a.reset();
            let a = runner_a.apply_circuit(&mut state_a, &commands).unwrap();
            state_b.reset();
            let b = runner_b.apply_circuit(&mut state_b, &commands).unwrap();
            assert_eq!(
                a.get_bit(QubitId(0)),
                b.get_bit(QubitId(0)),
                "general_noise() and GeneralNoiseModelBuilder::new() should be equivalent"
            );
        }
    }

    #[test]
    fn test_simple_depolarizing() {
        let model = GeneralNoiseModelBuilder::new()
            .with_p1(0.01)
            .with_p2(0.02)
            .build();

        // Should have single-qubit and two-qubit channels
        assert_eq!(model.channel_count(), 2);
    }

    #[test]
    fn test_full_configuration() {
        let model = GeneralNoiseModelBuilder::new()
            .with_p_prep(0.001)
            .with_p_prep_leak_ratio(0.1) // Enable leakage potential
            .with_p1(0.01)
            .with_p2(0.02)
            .with_p_meas(0.03, 0.04)
            .with_p_idle_linear(0.0001)
            .with_leakage_scale(1.0)
            .build();

        // Leakage + Prep + 1Q + 2Q + Meas + Idle = 6 channels
        assert_eq!(model.channel_count(), 6);
    }

    #[test]
    fn test_noiseless_gates() {
        let model = GeneralNoiseModelBuilder::new()
            .with_p1(0.01)
            .with_noiseless_gate(GateType::I)
            .with_noiseless_gates(&[GateType::SX, GateType::SXdg])
            .build();

        assert!(model.context().is_noiseless(GateType::I));
        assert!(model.context().is_noiseless(GateType::SX));
        assert!(model.context().is_noiseless(GateType::SXdg));
        assert!(!model.context().is_noiseless(GateType::H));
    }

    #[test]
    fn test_crosstalk_configuration() {
        let model = GeneralNoiseModelBuilder::new()
            .with_p_meas_crosstalk(0.01, 0.05)
            .with_p_meas_crosstalk_transitions(CrosstalkTransitions::symmetric_with_leakage())
            .build();

        // Leakage + Crosstalk = 2 channels (transitions have leakage potential)
        assert_eq!(model.channel_count(), 2);
    }

    #[test]
    fn test_crosstalk_without_leakage() {
        let model = GeneralNoiseModelBuilder::new()
            .with_p_meas_crosstalk(0.01, 0.05)
            .with_p_meas_crosstalk_transitions(CrosstalkTransitions::flip_only())
            .build();

        // Just crosstalk channel (no leakage potential)
        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_time_scale_configuration() {
        let model = GeneralNoiseModelBuilder::new()
            .with_time_scale(TimeScale::NANOSECONDS)
            .with_p1(0.01)
            .build();

        // Time scale should be passed through to the model
        assert!(model.time_scale().is_some());
        assert!((model.time_scale().unwrap().to_seconds(1000.into()) - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn test_idle_t1_t2_configuration() {
        // T1=50us, T2=30us with nanosecond time units
        let model = GeneralNoiseModelBuilder::new()
            .with_time_scale(TimeScale::NANOSECONDS)
            .with_idle_t1_t2(50e-6, 30e-6)
            .build();

        // Should have created an idle channel
        assert_eq!(model.channel_count(), 1);
    }

    // ========================================================================
    // Mixed Channel Tests
    // ========================================================================

    #[test]
    fn test_general_builder_with_flow_channel() {
        use crate::noise::composite::{CompositeChannelBuilder, prelude::*};

        // Mix traditional channels with a composite channel
        let model = GeneralNoiseModelBuilder::new()
            .with_p1(0.001) // Traditional 1Q channel
            .with_p_meas(0.02, 0.03) // Traditional measurement channel
            .with_channel(
                // Flow 2Q channel
                CompositeChannelBuilder::two_qubit("custom_2q", seq![prob(0.01, pauli()),]),
            )
            .build();

        // 1Q + Meas + Custom 2Q = 3 channels
        assert_eq!(model.channel_count(), 3);
    }

    #[test]
    fn test_flow_builder_with_traditional_channel() {
        use crate::noise::composite::CompositeNoiseModelBuilder;

        // Mix composite channels with a traditional channel
        let model = CompositeNoiseModelBuilder::new()
            .with_p1(0.001) // Flow 1Q channel
            .with_p2(0.01) // Flow 2Q channel
            .with_channel(MeasurementChannel::symmetric(0.02)) // Traditional
            .build();

        // 1Q + 2Q + Meas = 3 channels
        assert_eq!(model.channel_count(), 3);
    }

    #[test]
    fn test_mixed_channels_execution() {
        use crate::command::CommandBuilder;
        use crate::noise::composite::{CompositeChannelBuilder, prelude::*};
        use crate::runner::CircuitRunner;
        use pecos_qsim::SparseStab;

        // Create a model with both channel types
        let model = GeneralNoiseModelBuilder::new()
            .with_p1(0.0) // No traditional 1Q noise
            .with_channel(
                // But use composite for 2Q
                CompositeChannelBuilder::two_qubit("flow_2q", prob(0.5, pauli())),
            )
            .build();

        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .cx(0, 1)
            .mz(0)
            .mz(1)
            .build();

        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(model)
            .with_seed(42);

        // Should run without errors
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        assert_eq!(outcomes.len(), 2);
    }

    // ========================================================================
    // Builder Parity Tests (GeneralNoiseModelBuilder vs CompositeNoiseModelBuilder)
    // ========================================================================

    #[test]
    fn test_general_vs_flow_builder_single_qubit_parity() {
        use crate::command::CommandBuilder;
        use crate::noise::composite::CompositeNoiseModelBuilder;
        use crate::runner::CircuitRunner;
        use pecos_core::QubitId;
        use pecos_qsim::SparseStab;

        let p1 = 0.3; // High error rate for clear statistical signal
        let shots = 1000;

        // Build commands once
        let commands = CommandBuilder::new()
            .pz(0)
            .identity(0) // Identity gate (gets noise)
            .mz(0)
            .build();

        // Run with GeneralNoiseModelBuilder - count Z basis measurements
        let mut state = SparseStab::new(1);
        let mut general_ones = 0;
        for seed in 0..shots {
            let model = GeneralNoiseModelBuilder::new().with_p1(p1).build();

            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model)
                .with_seed(seed);

            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            if outcomes.get(QubitId(0)).is_some_and(|o| o.outcome) {
                general_ones += 1;
            }
        }

        // Run with CompositeNoiseModelBuilder
        let mut flow_ones = 0;
        for seed in 0..shots {
            let model = CompositeNoiseModelBuilder::new().with_p1(p1).build();

            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model)
                .with_seed(seed);

            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            if outcomes.get(QubitId(0)).is_some_and(|o| o.outcome) {
                flow_ones += 1;
            }
        }

        // Both should have similar error rates
        // With depolarizing noise, roughly 2/3 of errors cause bit flip (X or Y)
        // So expected ones ~ p1 * 2/3
        let expected_ones = (p1 * 2.0 / 3.0 * shots as f64) as i64;
        let tolerance = (0.2 * expected_ones as f64).max(50.0) as i64;

        // The two builders should produce similar error rates
        let rate_diff = (i64::from(general_ones) - i64::from(flow_ones)).abs();
        assert!(
            rate_diff < tolerance,
            "Builders differ too much: general_ones={general_ones}, flow_ones={flow_ones}, diff={rate_diff}, tolerance={tolerance}"
        );
    }

    #[test]
    fn test_general_vs_flow_builder_two_qubit_parity() {
        use crate::command::CommandBuilder;
        use crate::noise::composite::CompositeNoiseModelBuilder;
        use crate::runner::CircuitRunner;
        use pecos_core::QubitId;
        use pecos_qsim::SparseStab;

        let p2 = 0.3;
        let shots = 1000;

        // Build commands once
        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .cx(0, 1)
            .mz(0)
            .mz(1)
            .build();

        // Run with GeneralNoiseModelBuilder
        let mut state = SparseStab::new(2);
        let mut general_errors = 0;
        for seed in 0..shots {
            let model = GeneralNoiseModelBuilder::new().with_p2(p2).build();

            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model)
                .with_seed(seed);

            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            // Count if either qubit measured 1 (indicating error)
            let q0 = outcomes.get(QubitId(0)).is_some_and(|o| o.outcome);
            let q1 = outcomes.get(QubitId(1)).is_some_and(|o| o.outcome);
            if q0 || q1 {
                general_errors += 1;
            }
        }

        // Run with CompositeNoiseModelBuilder
        let mut composite_errors = 0;
        for seed in 0..shots {
            let model = CompositeNoiseModelBuilder::new().with_p2(p2).build();

            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model)
                .with_seed(seed);

            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            let q0 = outcomes.get(QubitId(0)).is_some_and(|o| o.outcome);
            let q1 = outcomes.get(QubitId(1)).is_some_and(|o| o.outcome);
            if q0 || q1 {
                composite_errors += 1;
            }
        }

        // Both should have similar error rates
        let rate_diff = (i64::from(general_errors) - i64::from(composite_errors)).abs();
        let tolerance = (0.15 * shots as f64) as i64;

        assert!(
            rate_diff < tolerance,
            "2Q builders differ too much: general={general_errors}, composite={composite_errors}, diff={rate_diff}, tolerance={tolerance}"
        );
    }

    #[test]
    fn test_general_vs_flow_builder_measurement_parity() {
        use crate::command::CommandBuilder;
        use crate::noise::composite::CompositeNoiseModelBuilder;
        use crate::runner::CircuitRunner;
        use pecos_core::QubitId;
        use pecos_qsim::SparseStab;

        let p_meas = 0.2;
        let shots = 1000;

        // Build commands once - Prepare |0> and measure
        let commands = CommandBuilder::new().pz(0).mz(0).build();

        // Run with GeneralNoiseModelBuilder - errors should flip to 1
        let mut state = SparseStab::new(1);
        let mut general_ones = 0;
        for seed in 0..shots {
            let model = GeneralNoiseModelBuilder::new()
                .with_p_meas_symmetric(p_meas)
                .build();

            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model)
                .with_seed(seed);

            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            if outcomes.get(QubitId(0)).is_some_and(|o| o.outcome) {
                general_ones += 1;
            }
        }

        // Run with CompositeNoiseModelBuilder
        let mut flow_ones = 0;
        for seed in 0..shots {
            let model = CompositeNoiseModelBuilder::new()
                .with_p_meas_symmetric(p_meas)
                .build();

            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model)
                .with_seed(seed);

            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            if outcomes.get(QubitId(0)).is_some_and(|o| o.outcome) {
                flow_ones += 1;
            }
        }

        // Both should have approximately p_meas flips
        let expected = (p_meas * shots as f64) as i64;
        let tolerance = (0.2 * expected as f64).max(50.0) as i64;

        assert!(
            (i64::from(general_ones) - expected).abs() < tolerance,
            "General measurement error rate off: expected ~{expected}, got {general_ones}"
        );

        assert!(
            (i64::from(flow_ones) - expected).abs() < tolerance,
            "Flow measurement error rate off: expected ~{expected}, got {flow_ones}"
        );
    }
}
