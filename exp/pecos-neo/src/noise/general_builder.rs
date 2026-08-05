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
use super::measurement::{MeasurementChannel, MeasurementStateFlipChannel};
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
use std::collections::BTreeMap;

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
    idle_after_2q: f64,

    // Measurement
    p_meas_0: f64,
    p_meas_1: f64,
    p_meas_state_flip: f64,
    p_meas_crosstalk_global: f64,
    p_meas_crosstalk_local: f64,
    p_meas_crosstalk_transitions: Option<CrosstalkTransitions>,

    // Idle noise
    p_idle_linear_rate: f64,
    p_idle_linear_weights: PauliWeights,
    p_idle_quadratic_rate: f64,
    p_idle_quadratic_configured: bool,
    p_idle_sin_squared: Option<(f64, BTreeMap<String, f64>)>,
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
            idle_after_2q: 0.0,

            // Measurement
            p_meas_0: 0.0,
            p_meas_1: 0.0,
            p_meas_state_flip: 0.0,
            p_meas_crosstalk_global: 0.0,
            p_meas_crosstalk_local: 0.0,
            p_meas_crosstalk_transitions: None,

            // Idle
            p_idle_linear_rate: 0.0,
            p_idle_linear_weights: PauliWeights::custom(0.0, 0.0, 1.0), // Z-only
            p_idle_quadratic_rate: 0.0,
            p_idle_quadratic_configured: false,
            p_idle_sin_squared: None,
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

    /// Set the duration of the idle-noise site applied after each two-qubit gate.
    ///
    /// A duration of zero disables these sites. Nonzero sites receive all
    /// configured linear and quadratic idle mechanisms.
    #[must_use]
    pub fn with_idle_after_2q(mut self, duration: f64) -> Self {
        self.idle_after_2q = duration;
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

    /// Set a measurement error realized as a physical X flip of the
    /// qubit state just before readout (engines depolarizing / DEM
    /// convention). Unlike `with_p_meas_symmetric`, the error persists
    /// in the post-measurement state: measuring the same qubit twice
    /// without a reset sees the second outcome flipped at `2p(1-p)`.
    #[must_use]
    pub fn with_p_meas_state_flip(mut self, p: f64) -> Self {
        self.p_meas_state_flip = p;
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

    /// Set the DEM-style linear idle-noise family.
    ///
    /// `rate` is the total event rate per time unit. For an idle of duration `d`, one event is
    /// sampled with probability `rate * d`, then its X, Y, or Z axis is drawn from `model`. The
    /// model must therefore be a normalized distribution: this linear family splits one total
    /// rate across its axes.
    ///
    /// In contrast, [`Self::with_p_idle_sin_squared`] takes radians per time unit and an
    /// unnormalized model because sine laws do not add linearly: each axis carries its own
    /// independent rate. That setter applies no `2*pi` conversion and no
    /// `coherent_to_incoherent_factor`, unlike [`Self::with_p_idle_quadratic`].
    ///
    /// Neo's linear family stores its model in [`PauliWeights`], so it cannot represent the DEM's
    /// L axis. An L key is rejected; use neo's [`LeakageChannel`] for linear leakage. The new
    /// sine-squared family uses separate map storage and accepts X, Y, Z, and L.
    ///
    /// All neo idle-noise families are off by default, so translating a DEM configuration only
    /// requires setting the requested families.
    ///
    /// The linear sampling structure deliberately remains different from the DEM: neo emits at
    /// most one linear event followed by a categorical axis choice, while the DEM emits independent
    /// per-axis mechanisms. The difference is second order in the rates; this setter aligns the
    /// units and axis alphabet that neo can represent, not that sampling structure.
    ///
    /// # Panics
    ///
    /// Panics if `rate` or a model value is not finite and non-negative, if the model is not
    /// normalized, or if it contains a key other than X, Y, or Z. L is rejected with guidance to
    /// use [`LeakageChannel`].
    #[must_use]
    pub fn with_p_idle_linear(mut self, rate: f64, model: &BTreeMap<String, f64>) -> Self {
        self.p_idle_linear_rate = Self::validate_finite_non_negative(rate, "linear idling rate");
        self.p_idle_linear_weights = Self::validate_linear_model(model);
        self
    }

    /// Set the quadratic idle noise rate (per time unit).
    ///
    /// The rate interpretation depends on your `TimeScale` configuration.
    #[must_use]
    pub fn with_p_idle_quadratic(mut self, rate: f64) -> Self {
        self.p_idle_quadratic_rate = rate;
        self.p_idle_quadratic_configured = true;
        self
    }

    /// Set the DEM-style stochastic sine-squared idle-noise family.
    ///
    /// `rate` is in radians per time unit. No `2*pi` conversion and no
    /// `coherent_to_incoherent_factor` is applied, unlike [`Self::with_p_idle_quadratic`]. For each
    /// axis P with multiplier `n_P` and an idle of duration `d`, neo independently samples
    /// `P(P) = sin^2(rate * n_P * d)`.
    ///
    /// The model accepts X, Y, Z, and L and is intentionally unnormalized: sine laws do not add
    /// linearly, so every axis carries its own independent rate. By comparison,
    /// [`Self::with_p_idle_linear`] requires a normalized distribution because its one total
    /// linear event rate is split across axes.
    ///
    /// Unlike the linear family's [`PauliWeights`] storage, this family has separate map storage
    /// that can represent the DEM's L axis. Sine-family leakage is tracked by neo and enables its
    /// [`LeakageChannel`].
    ///
    /// The legacy quadratic spelling has a different unit contract and folds
    /// `coherent_to_incoherent_factor` and the exact `sin^2(theta/2)` Pauli twirl into its
    /// stochastic path; this setter is the direct radians-per-time-unit spelling.
    ///
    /// All neo idle-noise families are off by default, so translating a DEM configuration only
    /// requires setting the requested families.
    ///
    /// # Panics
    ///
    /// Panics if `rate` or a multiplier is not finite and non-negative, or if `model` contains a
    /// key other than X, Y, Z, or L.
    #[must_use]
    pub fn with_p_idle_sin_squared(mut self, rate: f64, model: &BTreeMap<String, f64>) -> Self {
        let rate = Self::validate_finite_non_negative(rate, "sine-squared idling rate");
        Self::validate_sine_model(model);
        self.p_idle_sin_squared = Some((rate, model.clone()));
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
        self.p_idle_quadratic_configured = true;
        self
    }

    // ========================================================================
    // Build
    // ========================================================================

    /// Validate that a value is finite and non-negative.
    fn validate_finite_non_negative(value: f64, name: &str) -> f64 {
        assert!(
            value.is_finite() && value >= 0.0,
            "{name} must be finite and non-negative, got {value}"
        );
        value
    }

    /// Validate and convert a normalized X/Y/Z linear-family model.
    fn validate_linear_model(model: &BTreeMap<String, f64>) -> PauliWeights {
        const NORMALIZATION_TOLERANCE: f64 = 1e-5;

        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;
        for (axis, weight) in model {
            match axis.as_str() {
                "X" => x = *weight,
                "Y" => y = *weight,
                "Z" => z = *weight,
                "L" => panic!(
                    "neo's idle linear family cannot represent leakage; use neo's \
                     LeakageChannel for linear leakage"
                ),
                _ => panic!("p_idle_linear model has invalid key '{axis}'; expected X, Y, Z, or L"),
            }
            Self::validate_finite_non_negative(
                *weight,
                &format!("p_idle_linear weight for '{axis}'"),
            );
        }

        let total = x + y + z;
        assert!(
            total.is_finite() && (total - 1.0).abs() <= NORMALIZATION_TOLERANCE,
            "p_idle_linear model weights must sum to 1.0 within tolerance \
             {NORMALIZATION_TOLERANCE}, got {total}"
        );
        PauliWeights::custom(x / total, y / total, z / total)
    }

    /// Validate an unnormalized sine-family multiplier model.
    fn validate_sine_model(model: &BTreeMap<String, f64>) {
        for (axis, multiplier) in model {
            assert!(
                matches!(axis.as_str(), "X" | "Y" | "Z" | "L"),
                "p_idle_sin_squared model has invalid key '{axis}'; expected X, Y, Z, or L"
            );
            Self::validate_finite_non_negative(
                *multiplier,
                &format!("p_idle_sin_squared multiplier for '{axis}'"),
            );
        }
    }

    /// Validate combinations whose interpretation would otherwise depend on silent precedence.
    ///
    /// # Errors
    ///
    /// Returns a description of the conflicting spellings and their incompatible semantics.
    pub fn validate_configuration(&self) -> Result<(), &'static str> {
        if self.p_idle_sin_squared.is_some() && self.p_idle_quadratic_configured {
            return Err(
                "with_p_idle_sin_squared cannot be combined with with_p_idle_quadratic: \
                 the spellings use different units; with_p_idle_sin_squared uses radians per \
                 time unit with no conversion, while with_p_idle_quadratic uses the legacy \
                 quadratic-rate units and applies coherent_to_incoherent_factor",
            );
        }
        if self.p_idle_sin_squared.is_some() && self.p_idle_coherent {
            return Err(
                "with_p_idle_sin_squared cannot be combined with with_p_idle_coherent(true): \
                 with_p_idle_sin_squared is stochastic by definition, while \
                 with_p_idle_coherent(true) selects the legacy coherent path",
            );
        }
        Ok(())
    }

    /// Check if any configured parameters can cause leakage.
    fn has_leakage_potential(&self) -> bool {
        self.p_prep_leak_ratio > 0.0
            || self.p1_emission_ratio > 0.0
            || self.p2_emission_ratio > 0.0
            || self
                .p_meas_crosstalk_transitions
                .as_ref()
                .is_some_and(|t| t.from_0_leak > 0.0 || t.from_1_leak > 0.0)
            || self
                .p_idle_sin_squared
                .as_ref()
                .is_some_and(|(rate, model)| {
                    *rate > 0.0 && model.get("L").is_some_and(|multiplier| *multiplier > 0.0)
                })
    }

    /// Build the configured noise model.
    ///
    /// Returns a [`ComposableNoiseModel`] with all the configured channels.
    ///
    /// # Panics
    ///
    /// Panics if sine-squared idle noise is combined with the legacy quadratic or coherent idle
    /// path.
    #[must_use]
    pub fn build(self) -> ComposableNoiseModel {
        self.validate_configuration()
            .unwrap_or_else(|message| panic!("{message}"));

        let (p_idle_sin_squared_rate, p_idle_sin_squared_model) = self
            .p_idle_sin_squared
            .clone()
            .unwrap_or_else(|| (0.0, BTreeMap::new()));
        let mut model = ComposableNoiseModel::new().add_plugin(&CorePlugin);

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
            );
            model = model.add_channel(channel);
        }

        // Measurement channel
        if self.p_meas_state_flip > 0.0 {
            model = model.add_channel(MeasurementStateFlipChannel::new(self.p_meas_state_flip));
        }
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
        if self.p_idle_linear_rate > 0.0
            || self.p_idle_quadratic_rate > 0.0
            || p_idle_sin_squared_rate > 0.0
            || self.idle_after_2q > 0.0
        {
            let channel = IdleChannel {
                linear_rate: self.p_idle_linear_rate,
                linear_weights: self.p_idle_linear_weights,
                sin_squared_rate: p_idle_sin_squared_rate,
                sin_squared_model: p_idle_sin_squared_model,
                quadratic_rate: self.p_idle_quadratic_rate,
                coherent_dephasing: self.p_idle_coherent,
                coherent_to_incoherent_factor: self.p_idle_coherent_to_incoherent_factor,
                idle_after_2q: self.idle_after_2q,
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
#[allow(clippy::cast_precision_loss)] // statistical tests use count as f64
mod tests {
    use super::*;
    use crate::command::GateCommand;
    use crate::noise::{NoiseEvent, NoiseResponse};
    use pecos_core::QubitId;
    use pecos_random::PecosRng;

    fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
        if let Some(message) = panic.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = panic.downcast_ref::<&str>() {
            (*message).to_string()
        } else {
            "non-string panic".to_string()
        }
    }

    fn collect_gates(response: NoiseResponse) -> Vec<GateCommand> {
        match response {
            NoiseResponse::InjectGates(gates) => (*gates).into_vec(),
            NoiseResponse::Multiple(responses) => {
                responses.into_iter().flat_map(collect_gates).collect()
            }
            _ => Vec::new(),
        }
    }

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
        use pecos_simulators::SparseStab;

        let commands = CommandBuilder::new().pz(&[0]).z(&[0]).mz(&[0]).build();

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
    fn after_2q_idle_works_without_p2_or_a_two_qubit_channel() {
        let linear_model = BTreeMap::from([("X".to_string(), 1.0)]);
        let mut model = GeneralNoiseModelBuilder::new()
            .with_p_idle_linear(1.0, &linear_model)
            .with_idle_after_2q(1.0)
            .build();
        assert_eq!(model.channel_names(), ["IdleChannel"]);

        let qubits = [QubitId(0), QubitId(1)];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &[],
            gate_id: None,
        };
        let gates = collect_gates(model.emit(&event, &mut PecosRng::seed_from_u64(67)));

        assert_eq!(gates.len(), 2);
        assert!(gates.iter().all(|gate| gate.gate_type == GateType::X));
    }

    #[test]
    fn quadratic_only_after_2q_configuration_builds_and_emits() {
        let mut model = GeneralNoiseModelBuilder::new()
            .with_p_idle_quadratic(std::f64::consts::PI)
            .with_idle_after_2q(1.0)
            .build();
        assert_eq!(model.channel_names(), ["IdleChannel"]);

        let qubits = [QubitId(0), QubitId(1)];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &[],
            gate_id: None,
        };
        let gates = collect_gates(model.emit(&event, &mut PecosRng::seed_from_u64(71)));

        assert_eq!(gates.len(), 2);
        assert!(gates.iter().all(|gate| gate.gate_type == GateType::Z));
    }

    #[test]
    fn sine_family_reaches_after_2q_idle_sites() {
        let sine_model = BTreeMap::from([("X".to_string(), 1.0)]);
        let mut model = GeneralNoiseModelBuilder::new()
            .with_p_idle_sin_squared(std::f64::consts::FRAC_PI_2, &sine_model)
            .with_idle_after_2q(1.0)
            .build();
        let qubits = [QubitId(0), QubitId(1)];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &[],
            gate_id: None,
        };
        let gates = collect_gates(model.emit(&event, &mut PecosRng::seed_from_u64(73)));

        assert_eq!(gates.len(), 2);
        assert!(gates.iter().all(|gate| gate.gate_type == GateType::X));
    }

    #[test]
    fn sine_multipliers_are_not_normalized_by_the_builder() {
        let sine_model = BTreeMap::from([("X".to_string(), 1.0), ("Z".to_string(), 1.0)]);
        let mut model = GeneralNoiseModelBuilder::new()
            .with_p_idle_sin_squared(std::f64::consts::FRAC_PI_2, &sine_model)
            .build();
        let qubits = std::array::from_fn::<_, 16, _>(QubitId);
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration: 1.into(),
        };
        let gates = collect_gates(model.emit(&event, &mut PecosRng::seed_from_u64(79)));

        assert_eq!(gates.len(), 32);
        assert!(gates[..16].iter().all(|gate| gate.gate_type == GateType::X));
        assert!(gates[16..].iter().all(|gate| gate.gate_type == GateType::Z));
    }

    #[test]
    fn sine_family_accepts_leakage_and_enables_leakage_channel() {
        let sine_model = BTreeMap::from([("L".to_string(), 1.0)]);
        let mut model = GeneralNoiseModelBuilder::new()
            .with_p_idle_sin_squared(std::f64::consts::FRAC_PI_2, &sine_model)
            .build();
        assert_eq!(model.channel_names(), ["LeakageChannel", "IdleChannel"]);

        let qubits = [QubitId(0)];
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration: 1.into(),
        };
        let response = model.emit(&event, &mut PecosRng::seed_from_u64(83));

        assert!(matches!(response, NoiseResponse::MarkLeaked(_)));
        assert!(model.context().is_leaked(QubitId(0)));
    }

    #[test]
    fn linear_family_rejects_unnormalized_model() {
        let linear_model = BTreeMap::from([("X".to_string(), 1.0), ("Z".to_string(), 1.0)]);
        let panic = std::panic::catch_unwind(|| {
            let _ = GeneralNoiseModelBuilder::new().with_p_idle_linear(0.1, &linear_model);
        })
        .unwrap_err();

        assert!(panic_message(panic.as_ref()).contains("must sum to 1.0"));
    }

    #[test]
    fn linear_family_rejects_leakage_with_neo_guidance() {
        let linear_model = BTreeMap::from([("X".to_string(), 0.5), ("L".to_string(), 0.5)]);
        let panic = std::panic::catch_unwind(|| {
            let _ = GeneralNoiseModelBuilder::new().with_p_idle_linear(0.1, &linear_model);
        })
        .unwrap_err();
        let message = panic_message(panic.as_ref());

        assert!(message.contains("neo's idle linear family cannot represent leakage"));
        assert!(message.contains("neo's LeakageChannel"));
    }

    #[test]
    fn linear_family_rejects_invalid_rates_axes_and_weights() {
        let normalized = BTreeMap::from([("X".to_string(), 1.0)]);
        for rate in [f64::INFINITY, f64::NAN, -0.1] {
            let panic = std::panic::catch_unwind(|| {
                let _ = GeneralNoiseModelBuilder::new().with_p_idle_linear(rate, &normalized);
            })
            .unwrap_err();
            assert!(panic_message(panic.as_ref()).contains("finite and non-negative"));
        }

        let invalid_axis = BTreeMap::from([("A".to_string(), 1.0)]);
        let panic = std::panic::catch_unwind(|| {
            let _ = GeneralNoiseModelBuilder::new().with_p_idle_linear(0.1, &invalid_axis);
        })
        .unwrap_err();
        assert!(panic_message(panic.as_ref()).contains("invalid key 'A'"));

        for invalid_weight in [f64::INFINITY, f64::NAN, -1.0] {
            let invalid_model = BTreeMap::from([
                ("X".to_string(), invalid_weight),
                ("Z".to_string(), 1.0 - invalid_weight),
            ]);
            let panic = std::panic::catch_unwind(|| {
                let _ = GeneralNoiseModelBuilder::new().with_p_idle_linear(0.1, &invalid_model);
            })
            .unwrap_err();
            assert!(panic_message(panic.as_ref()).contains("finite and non-negative"));
        }
    }

    #[test]
    fn sine_family_rejects_invalid_rates_axes_and_multipliers() {
        let valid_model = BTreeMap::from([("X".to_string(), 1.0)]);
        for rate in [f64::INFINITY, f64::NAN, -0.1] {
            let panic = std::panic::catch_unwind(|| {
                let _ = GeneralNoiseModelBuilder::new().with_p_idle_sin_squared(rate, &valid_model);
            })
            .unwrap_err();
            assert!(panic_message(panic.as_ref()).contains("finite and non-negative"));
        }

        let invalid_axis = BTreeMap::from([("A".to_string(), 1.0)]);
        let panic = std::panic::catch_unwind(|| {
            let _ = GeneralNoiseModelBuilder::new().with_p_idle_sin_squared(0.1, &invalid_axis);
        })
        .unwrap_err();
        assert!(panic_message(panic.as_ref()).contains("invalid key 'A'"));

        for multiplier in [f64::INFINITY, f64::NAN, -1.0] {
            let invalid_model = BTreeMap::from([("X".to_string(), multiplier)]);
            let panic = std::panic::catch_unwind(|| {
                let _ =
                    GeneralNoiseModelBuilder::new().with_p_idle_sin_squared(0.1, &invalid_model);
            })
            .unwrap_err();
            assert!(panic_message(panic.as_ref()).contains("finite and non-negative"));
        }
    }

    #[test]
    fn sine_family_conflicts_with_legacy_quadratic_spelling() {
        let sine_model = BTreeMap::from([("Z".to_string(), 1.0)]);
        let builder = GeneralNoiseModelBuilder::new()
            .with_p_idle_quadratic(0.1)
            .with_p_idle_sin_squared(0.1, &sine_model);
        let error = builder.validate_configuration().unwrap_err();

        assert!(error.contains("with_p_idle_sin_squared"));
        assert!(error.contains("with_p_idle_quadratic"));
        assert!(error.contains("different units"));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| builder.build())).is_err()
        );
    }

    #[test]
    fn sine_family_conflicts_with_coherent_legacy_path() {
        let sine_model = BTreeMap::from([("Z".to_string(), 1.0)]);
        let builder = GeneralNoiseModelBuilder::new()
            .with_p_idle_sin_squared(0.1, &sine_model)
            .with_p_idle_coherent(true);
        let error = builder.validate_configuration().unwrap_err();

        assert!(error.contains("with_p_idle_sin_squared"));
        assert!(error.contains("with_p_idle_coherent(true)"));
        assert!(error.contains("stochastic by definition"));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| builder.build())).is_err()
        );
    }

    #[test]
    fn test_full_configuration() {
        let linear_model = BTreeMap::from([("Z".to_string(), 1.0)]);
        let model = GeneralNoiseModelBuilder::new()
            .with_p_prep(0.001)
            .with_p_prep_leak_ratio(0.1) // Enable leakage potential
            .with_p1(0.01)
            .with_p2(0.02)
            .with_p_meas(0.03, 0.04)
            .with_p_idle_linear(0.0001, &linear_model)
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
        use pecos_simulators::SparseStab;

        // Create a model with both channel types
        let model = GeneralNoiseModelBuilder::new()
            .with_p1(0.0) // No traditional 1Q noise
            .with_channel(
                // But use composite for 2Q
                CompositeChannelBuilder::two_qubit("flow_2q", prob(0.5, pauli())),
            )
            .build();

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

        // Should run without errors
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        assert_eq!(outcomes.len(), 2);
    }

    // ========================================================================
    // Builder Parity Tests (GeneralNoiseModelBuilder vs CompositeNoiseModelBuilder)
    // ========================================================================

    #[test]
    #[allow(clippy::cast_possible_truncation)] // test statistical bounds
    fn test_general_vs_flow_builder_single_qubit_parity() {
        use crate::command::CommandBuilder;
        use crate::noise::composite::CompositeNoiseModelBuilder;
        use crate::runner::CircuitRunner;
        use pecos_core::QubitId;
        use pecos_simulators::SparseStab;

        let p1 = 0.3; // High error rate for clear statistical signal
        let shots = 1000;

        // Build commands once
        let commands = CommandBuilder::new()
            .pz(&[0])
            .identity(&[0]) // Identity gate (gets noise)
            .mz(&[0])
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
    #[allow(clippy::cast_possible_truncation)] // test statistical bounds
    fn test_general_vs_flow_builder_two_qubit_parity() {
        use crate::command::CommandBuilder;
        use crate::noise::composite::CompositeNoiseModelBuilder;
        use crate::runner::CircuitRunner;
        use pecos_core::QubitId;
        use pecos_simulators::SparseStab;

        let p2 = 0.3;
        let shots = 1000;

        // Build commands once
        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .cx(&[(0, 1)])
            .mz(&[0])
            .mz(&[1])
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
    #[allow(clippy::cast_possible_truncation)] // test statistical bounds
    fn test_general_vs_flow_builder_measurement_parity() {
        use crate::command::CommandBuilder;
        use crate::noise::composite::CompositeNoiseModelBuilder;
        use crate::runner::CircuitRunner;
        use pecos_core::QubitId;
        use pecos_simulators::SparseStab;

        let p_meas = 0.2;
        let shots = 1000;

        // Build commands once - Prepare |0> and measure
        let commands = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

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
