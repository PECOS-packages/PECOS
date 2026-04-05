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

//! Unified noise model builder.
//!
//! This module provides [`NoiseModelBuilder`], the primary way to construct noise models.
//! It unifies simple parameter-based configuration with composable channel construction.
//!
//! # Philosophy
//!
//! The noise system is built on a few key concepts:
//!
//! - **Base channels**: Atomic noise operations (depolarizing, amplitude damping, etc.)
//! - **Composition**: Combining channels with logic (probability, conditions, sequences)
//! - **Events**: All noise is event-driven (after gates, measurements, etc.)
//!
//! This builder lets you work at whatever level of abstraction you need:
//!
//! ```no_run
//! use pecos_neo::noise::prelude::*;
//!
//! // Simple: just set error rates
//! let simple = NoiseModelBuilder::new()
//!     .with_depolarizing(0.001, 0.01)
//!     .with_measurement_error(0.02)
//!     .build();
//!
//! // Composed: build custom decision trees
//! let composed = NoiseModelBuilder::new()
//!     .with_single_qubit_noise(seq![
//!         skip_if_leaked(),
//!         prob(0.001, when_leaked(seep(), pauli())),
//!     ])
//!     .with_two_qubit_noise(seq![
//!         skip_if_leaked(),
//!         prob(0.01, two_qubit_pauli()),
//!     ])
//!     .build();
//!
//! // Mixed: combine both approaches
//! let mixed = NoiseModelBuilder::new()
//!     .with_depolarizing(0.001, 0.01)  // Simple base rates
//!     .with_channel(LeakageChannel::new())  // Add custom channel
//!     .build();
//! ```

use super::composite::Primitive;
use super::composite::channel::{CompositeChannel, CompositeChannelBuilder};
use super::crosstalk::CrosstalkChannel;
use super::idle::IdleChannel;
use super::leakage::LeakageChannel;
use super::measurement::MeasurementChannel;
use super::plugins::CorePlugin;
use super::preparation::PreparationChannel;
use super::single_qubit::SingleQubitChannel;
use super::two_qubit::{AngleScaling, TwoQubitChannel};
use super::{
    ComposableNoiseModel, CrosstalkTransitions, NoiseChannel, PauliWeights,
    SingleQubitEmissionWeights, TwoQubitEmissionWeights, TwoQubitPauliWeights,
};
use crate::command::GateType;
use pecos_core::TimeScale;

/// Unified builder for constructing noise models.
///
/// This is the primary way to build noise models in pecos-neo. It supports:
///
/// - **Simple configuration**: Set error rates and let the builder create appropriate channels
/// - **Custom channels**: Add pre-built channels directly
/// - **Composed channels**: Build decision trees using primitives (prob, when, seq, etc.)
///
/// # Examples
///
/// ## Simple depolarizing noise
///
/// ```
/// use pecos_neo::noise::prelude::*;
///
/// let model = NoiseModelBuilder::new()
///     .with_depolarizing(0.001, 0.01)  // p1=0.001, p2=0.01
///     .build();
/// ```
///
/// ## Custom composed noise
///
/// ```
/// use pecos_neo::noise::prelude::*;
///
/// let model = NoiseModelBuilder::new()
///     .with_single_qubit_noise(seq![
///         skip_if_leaked(),
///         prob(0.001, pauli()),
///     ])
///     .build();
/// ```
///
/// ## Adding existing channels
///
/// ```
/// use pecos_neo::noise::prelude::*;
///
/// let custom = SingleQubitChannel::depolarizing(0.001);
/// let model = NoiseModelBuilder::new()
///     .with_channel(custom)
///     .build();
/// ```
#[allow(clippy::struct_excessive_bools)]
pub struct NoiseModelBuilder {
    // ========================================================================
    // Simple parameter-based configuration
    // ========================================================================

    // Preparation
    p_prep: f64,
    p_prep_leak_ratio: f64,

    // Single-qubit gates (simple mode)
    p1: f64,
    p1_emission_ratio: f64,
    p1_emission_weights: SingleQubitEmissionWeights,
    p1_pauli_weights: PauliWeights,
    p1_seepage_prob: f64,

    // Two-qubit gates (simple mode)
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
    p_idle_coherent_factor: f64,

    // Leakage
    leakage_scale: f64,

    // Configuration
    noiseless_gates: Vec<GateType>,
    time_scale: Option<TimeScale>,

    // ========================================================================
    // Custom channels (composed or pre-built)
    // ========================================================================
    custom_channels: Vec<Box<dyn NoiseChannel>>,

    // Track if simple mode channels were overridden
    single_qubit_override: bool,
    two_qubit_override: bool,
    measurement_override: bool,
    preparation_override: bool,
}

impl Default for NoiseModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseModelBuilder {
    /// Create a new builder with all parameters set to zero/default.
    #[must_use]
    pub fn new() -> Self {
        Self {
            // Preparation
            p_prep: 0.0,
            p_prep_leak_ratio: 0.0,

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
            p_idle_linear_weights: PauliWeights::uniform(),
            p_idle_quadratic_rate: 0.0,
            p_idle_coherent: false,
            p_idle_coherent_factor: 1.0,

            // Leakage
            leakage_scale: 1.0,

            // Configuration
            noiseless_gates: Vec::new(),
            time_scale: None,

            // Custom
            custom_channels: Vec::new(),
            single_qubit_override: false,
            two_qubit_override: false,
            measurement_override: false,
            preparation_override: false,
        }
    }

    // ========================================================================
    // Simple Configuration (parameter-based)
    // ========================================================================

    /// Set depolarizing error rates for single-qubit (p1) and two-qubit (p2) gates.
    ///
    /// This is the simplest way to add gate noise. For more control, use
    /// `with_single_qubit_noise` or `with_two_qubit_noise`.
    #[must_use]
    pub fn with_depolarizing(mut self, p1: f64, p2: f64) -> Self {
        self.p1 = p1;
        self.p2 = p2;
        self
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

    /// Set two-qubit gate error probability with angle scaling.
    #[must_use]
    pub fn with_p2_scaled(mut self, p2: f64, scaling: AngleScaling) -> Self {
        self.p2 = p2;
        self.p2_angle_scaling = scaling;
        self
    }

    /// Set symmetric measurement error probability.
    #[must_use]
    pub fn with_measurement_error(mut self, p_meas: f64) -> Self {
        self.p_meas_0 = p_meas;
        self.p_meas_1 = p_meas;
        self
    }

    /// Set asymmetric measurement error probabilities.
    ///
    /// - `p_meas_0`: probability of flipping a 0 to 1
    /// - `p_meas_1`: probability of flipping a 1 to 0
    #[must_use]
    pub fn with_measurement_error_asymmetric(mut self, p_meas_0: f64, p_meas_1: f64) -> Self {
        self.p_meas_0 = p_meas_0;
        self.p_meas_1 = p_meas_1;
        self
    }

    /// Set preparation error probability.
    #[must_use]
    pub fn with_preparation_error(mut self, p_prep: f64) -> Self {
        self.p_prep = p_prep;
        self
    }

    /// Set preparation error with leakage ratio.
    #[must_use]
    pub fn with_preparation_error_with_leakage(mut self, p_prep: f64, leak_ratio: f64) -> Self {
        self.p_prep = p_prep;
        self.p_prep_leak_ratio = leak_ratio;
        self
    }

    /// Set emission (leakage) ratio for single-qubit gates.
    #[must_use]
    pub fn with_p1_emission_ratio(mut self, ratio: f64) -> Self {
        self.p1_emission_ratio = ratio;
        self
    }

    /// Set seepage probability for single-qubit gates.
    #[must_use]
    pub fn with_p1_seepage(mut self, prob: f64) -> Self {
        self.p1_seepage_prob = prob;
        self
    }

    /// Set emission (leakage) ratio for two-qubit gates.
    #[must_use]
    pub fn with_p2_emission_ratio(mut self, ratio: f64) -> Self {
        self.p2_emission_ratio = ratio;
        self
    }

    /// Set seepage probability for two-qubit gates.
    #[must_use]
    pub fn with_p2_seepage(mut self, prob: f64) -> Self {
        self.p2_seepage_prob = prob;
        self
    }

    /// Set Pauli weights for single-qubit errors.
    #[must_use]
    pub fn with_p1_pauli_weights(mut self, weights: PauliWeights) -> Self {
        self.p1_pauli_weights = weights;
        self
    }

    /// Set Pauli weights for two-qubit errors.
    #[must_use]
    pub fn with_p2_pauli_weights(mut self, weights: TwoQubitPauliWeights) -> Self {
        self.p2_pauli_weights = weights;
        self
    }

    /// Set idle noise parameters.
    #[must_use]
    pub fn with_idle_noise(mut self, linear_rate: f64, quadratic_rate: f64) -> Self {
        self.p_idle_linear_rate = linear_rate;
        self.p_idle_quadratic_rate = quadratic_rate;
        self
    }

    /// Enable coherent idle noise.
    #[must_use]
    pub fn with_coherent_idle(mut self, factor: f64) -> Self {
        self.p_idle_coherent = true;
        self.p_idle_coherent_factor = factor;
        self
    }

    /// Set time scale for physical time interpretation.
    #[must_use]
    pub fn with_time_scale(mut self, scale: TimeScale) -> Self {
        self.time_scale = Some(scale);
        self
    }

    /// Mark specific gate types as noiseless.
    #[must_use]
    pub fn with_noiseless_gate(mut self, gate_type: GateType) -> Self {
        self.noiseless_gates.push(gate_type);
        self
    }

    /// Mark multiple gate types as noiseless.
    #[must_use]
    pub fn with_noiseless_gates(mut self, gate_types: impl IntoIterator<Item = GateType>) -> Self {
        self.noiseless_gates.extend(gate_types);
        self
    }

    // ========================================================================
    // Composed Channel Configuration
    // ========================================================================

    /// Set custom single-qubit gate noise using a composed primitive.
    ///
    /// This overrides the simple p1-based configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::prelude::*;
    ///
    /// let model = NoiseModelBuilder::new()
    ///     .with_single_qubit_noise(seq![
    ///         skip_if_leaked(),
    ///         prob(0.001, when_leaked(seep(), pauli())),
    ///     ])
    ///     .build();
    /// ```
    #[must_use]
    pub fn with_single_qubit_noise<P: Primitive + Clone + 'static>(mut self, primitive: P) -> Self {
        let channel = CompositeChannelBuilder::single_qubit("single_qubit", primitive);
        self.custom_channels.push(Box::new(channel));
        self.single_qubit_override = true;
        self
    }

    /// Set custom two-qubit gate noise using a composed primitive.
    ///
    /// This overrides the simple p2-based configuration.
    #[must_use]
    pub fn with_two_qubit_noise<P: Primitive + Clone + 'static>(mut self, primitive: P) -> Self {
        let channel = CompositeChannelBuilder::two_qubit("two_qubit", primitive);
        self.custom_channels.push(Box::new(channel));
        self.two_qubit_override = true;
        self
    }

    /// Set custom measurement noise using a composed primitive.
    ///
    /// This overrides the simple p_meas-based configuration.
    #[must_use]
    pub fn with_measurement_noise<P: Primitive + Clone + 'static>(mut self, primitive: P) -> Self {
        let channel = CompositeChannelBuilder::after_measurement("measurement", primitive);
        self.custom_channels.push(Box::new(channel));
        self.measurement_override = true;
        self
    }

    /// Set custom preparation noise using a composed primitive.
    ///
    /// This overrides the simple p_prep-based configuration.
    #[must_use]
    pub fn with_preparation_noise<P: Primitive + Clone + 'static>(mut self, primitive: P) -> Self {
        let channel = CompositeChannelBuilder::preparation("preparation", primitive);
        self.custom_channels.push(Box::new(channel));
        self.preparation_override = true;
        self
    }

    /// Add custom noise for specific events using a composed primitive.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::prelude::*;
    ///
    /// let model = NoiseModelBuilder::new()
    ///     .with_custom_channel(
    ///         CompositeChannelBuilder::any_gate("custom_noise", seq![
    ///             skip_if_leaked(),
    ///             prob(0.05, pauli()),
    ///         ])
    ///     )
    ///     .build();
    /// ```
    #[must_use]
    pub fn with_custom_channel<P: Primitive + Clone + 'static>(
        mut self,
        channel: CompositeChannel<P>,
    ) -> Self {
        self.custom_channels.push(Box::new(channel));
        self
    }

    // ========================================================================
    // Direct Channel Addition
    // ========================================================================

    /// Add an existing channel to the noise model.
    ///
    /// Use this when you have a pre-built channel (traditional or composed).
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::prelude::*;
    ///
    /// let leakage = LeakageChannel::new();
    /// let model = NoiseModelBuilder::new()
    ///     .with_depolarizing(0.001, 0.01)
    ///     .with_channel(leakage)
    ///     .build();
    /// ```
    #[must_use]
    pub fn with_channel<C: NoiseChannel + 'static>(mut self, channel: C) -> Self {
        self.custom_channels.push(Box::new(channel));
        self
    }

    /// Add a boxed channel to the noise model.
    #[must_use]
    pub fn with_boxed_channel(mut self, channel: Box<dyn NoiseChannel>) -> Self {
        self.custom_channels.push(channel);
        self
    }

    // ========================================================================
    // Build
    // ========================================================================

    /// Build the noise model.
    #[must_use]
    pub fn build(self) -> ComposableNoiseModel {
        let mut model = ComposableNoiseModel::new().add_plugin(&CorePlugin);

        // Set time scale if provided
        if let Some(scale) = self.time_scale {
            model = model.with_time_scale(scale);
        }

        // Set noiseless gates
        for gate in &self.noiseless_gates {
            model.context_mut().add_noiseless_gate(*gate);
        }

        // Add single-qubit channel (unless overridden)
        if !self.single_qubit_override && self.p1 > 0.0 {
            let channel = SingleQubitChannel::new(
                self.p1,
                self.p1_pauli_weights,
                self.p1_emission_ratio,
                self.p1_emission_weights,
                self.p1_seepage_prob,
            );
            model = model.add_channel(channel);
        }

        // Add two-qubit channel (unless overridden)
        if !self.two_qubit_override && self.p2 > 0.0 {
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

        // Add measurement channel (unless overridden)
        if !self.measurement_override && (self.p_meas_0 > 0.0 || self.p_meas_1 > 0.0) {
            let channel = MeasurementChannel::asymmetric(self.p_meas_0, self.p_meas_1);
            model = model.add_channel(channel);

            // Add measurement crosstalk if configured
            if self.p_meas_crosstalk_global > 0.0 || self.p_meas_crosstalk_local > 0.0 {
                let mut crosstalk = CrosstalkChannel::new()
                    .with_global_rate(self.p_meas_crosstalk_global)
                    .with_local_rate(self.p_meas_crosstalk_local);
                if let Some(transitions) = self.p_meas_crosstalk_transitions {
                    crosstalk = crosstalk.with_transitions(transitions);
                }
                model = model.add_channel(crosstalk);
            }
        }

        // Add preparation channel (unless overridden)
        if !self.preparation_override && self.p_prep > 0.0 {
            let mut channel = PreparationChannel::new(self.p_prep);
            if self.p_prep_leak_ratio > 0.0 {
                channel = channel.with_leakage(self.p_prep_leak_ratio);
            }
            model = model.add_channel(channel);
        }

        // Add idle channel
        if self.p_idle_linear_rate > 0.0 {
            let mut channel = IdleChannel::linear(self.p_idle_linear_rate);
            if self.p_idle_coherent {
                channel = channel
                    .with_coherent_dephasing(true)
                    .with_coherent_to_incoherent_factor(self.p_idle_coherent_factor);
            } else {
                channel = channel.with_linear_weights(self.p_idle_linear_weights);
            }
            model = model.add_channel(channel);
        }

        // Add leakage channel (if scale differs from default of 1.0)
        if (self.leakage_scale - 1.0).abs() > f64::EPSILON {
            let channel = LeakageChannel::new().with_scale(self.leakage_scale);
            model = model.add_channel(channel);
        }

        // Add all custom channels
        for channel in self.custom_channels {
            model = model.add_boxed_channel(channel);
        }

        model
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)] // statistical tests use count as f64
mod tests {
    use super::*;
    use crate::command::CommandBuilder;
    use crate::noise::composite::prelude::*;
    use crate::runner::CircuitRunner;
    use pecos_core::QubitId;
    use pecos_simulators::SparseStab;

    #[test]
    fn test_simple_depolarizing() {
        let model = NoiseModelBuilder::new().with_depolarizing(0.1, 0.2).build();

        // Verify channels are added
        assert!(model.channel_count() >= 2);
    }

    #[test]
    fn test_composed_single_qubit() {
        let model = NoiseModelBuilder::new()
            .with_single_qubit_noise(prob(0.1, pauli()))
            .build();

        assert!(model.channel_count() >= 1);
    }

    #[test]
    fn test_mixed_configuration() {
        let model = NoiseModelBuilder::new()
            .with_p2(0.01)
            .with_single_qubit_noise(seq![skip_if_leaked(), prob(0.001, pauli())])
            .with_measurement_error(0.02)
            .build();

        // Should have: 2Q channel, custom 1Q channel, measurement channel, leakage
        assert!(model.channel_count() >= 3);
    }

    #[test]
    fn test_add_channel() {
        let custom = SingleQubitChannel::depolarizing(0.01);
        let model = NoiseModelBuilder::new().with_channel(custom).build();

        assert!(model.channel_count() >= 1);
    }

    #[test]
    fn test_composed_vs_simple_parity() {
        // Build the same noise model two ways and verify similar behavior
        let p1 = 0.1;
        let shots = 500;

        let commands = CommandBuilder::new()
            .pz(&[0])
            .h(&[0])
            .h(&[0])
            .mz(&[0])
            .build();

        // Simple configuration
        let mut state = SparseStab::new(1);
        let mut simple_errors = 0;
        for seed in 0..shots {
            let model = NoiseModelBuilder::new().with_p1(p1).build();
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model)
                .with_seed(seed);
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            if outcomes.get(QubitId(0)).is_none_or(|o| o.outcome) {
                simple_errors += 1;
            }
        }

        // Composed configuration
        let mut composed_errors = 0;
        for seed in 0..shots {
            let model = NoiseModelBuilder::new()
                .with_single_qubit_noise(prob(p1, pauli()))
                .build();
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model)
                .with_seed(seed);
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            if outcomes.get(QubitId(0)).is_none_or(|o| o.outcome) {
                composed_errors += 1;
            }
        }

        // Both should have similar error rates (within statistical tolerance)
        let simple_rate = f64::from(simple_errors) / shots as f64;
        let composed_rate = f64::from(composed_errors) / shots as f64;

        assert!(
            (simple_rate - composed_rate).abs() < 0.15,
            "Simple rate {simple_rate:.3} vs Composed rate {composed_rate:.3}"
        );
    }
}
