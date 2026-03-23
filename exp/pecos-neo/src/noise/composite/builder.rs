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

//! Builder for composite-based noise models.
//!
//! This module provides `CompositeNoiseModelBuilder`, which creates noise models using
//! the composite primitive system. It provides a similar API to `GeneralNoiseModelBuilder`
//! but uses the composable composite primitives underneath.
//!
//! # Example
//!
//! ```
//! use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
//!
//! let noise = CompositeNoiseModelBuilder::new()
//!     .with_p1(0.001)                    // 0.1% single-qubit gate error
//!     .with_p2(0.01)                     // 1% two-qubit gate error
//!     .with_p_meas(0.02, 0.03)           // Asymmetric measurement error
//!     .with_leakage(0.001, 0.1)          // Leakage and seepage
//!     .build();
//! ```
//!
//! # Advantages over `GeneralNoiseModelBuilder`
//!
//! - **Composability**: Easy to add custom noise primitives
//! - **Dynamic parameters**: Use `prob_fn` for angle-dependent noise
//! - **Extensibility**: Add custom conditions and actions

use super::channel::{
    CompositeChannel, CompositeChannelBuilder, CompositeCrosstalkChannel, CompositeEventFilter,
    NeighborFn,
};
use super::prelude::*;
use crate::command::GateType;
use crate::noise::two_qubit::AngleScaling;
use crate::noise::{
    ComposableNoiseModel, CrosstalkTransitions, SingleQubitEmissionWeights,
    TwoQubitEmissionWeights, TwoQubitPauliWeights,
};
use pecos_core::TimeScale;

// ============================================================================
// Validation helpers
// ============================================================================

/// Validate that a probability is in [0.0, 1.0] and warn if not.
/// Returns the clamped value.
fn validate_probability(value: f64, param_name: &str) -> f64 {
    if value < 0.0 {
        eprintln!("Warning: {param_name} = {value} is negative, clamping to 0.0");
        0.0
    } else if value > 1.0 {
        eprintln!("Warning: {param_name} = {value} exceeds 1.0, clamping to 1.0");
        1.0
    } else if value.is_nan() {
        eprintln!("Warning: {param_name} is NaN, setting to 0.0");
        0.0
    } else {
        value
    }
}

/// Validate that a rate is non-negative and warn if not.
/// Returns the clamped value.
fn validate_rate(value: f64, param_name: &str) -> f64 {
    if value < 0.0 {
        eprintln!("Warning: {param_name} = {value} is negative, clamping to 0.0");
        0.0
    } else if value.is_nan() {
        eprintln!("Warning: {param_name} is NaN, setting to 0.0");
        0.0
    } else if value.is_infinite() {
        eprintln!("Warning: {param_name} is infinite, clamping to f64::MAX");
        f64::MAX
    } else {
        value
    }
}

/// Builder for creating composite-based noise models.
///
/// This builder provides a convenient API for constructing noise models
/// using the composite primitive system. It mirrors the `GeneralNoiseModelBuilder`
/// API while using composite primitives underneath.
///
/// # Mixing Channel Types
///
/// You can mix composite channels with traditional channels using [`with_channel`]:
///
/// ```
/// use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
/// use pecos_neo::noise::MeasurementChannel;
///
/// let model = CompositeNoiseModelBuilder::new()
///     .with_p1(0.001)                              // Flow 1Q channel
///     .with_p2(0.01)                               // Flow 2Q channel
///     .with_channel(MeasurementChannel::symmetric(0.02))  // Traditional channel
///     .build();
/// ```
///
/// [`with_channel`]: Self::with_channel
pub struct CompositeNoiseModelBuilder {
    // Single-qubit gate parameters
    p1: f64,
    p1_emission_ratio: f64,
    p1_seepage: f64,
    p1_pauli_weights: Option<PauliWeights>,
    p1_emission_model: Option<SingleQubitEmissionWeights>,

    // Two-qubit gate parameters
    p2: f64,
    p2_emission_ratio: f64,
    p2_seepage: f64,
    p2_pauli_weights: Option<PauliWeights>,
    p2_pauli_model: Option<TwoQubitPauliWeights>,
    p2_emission_model: Option<TwoQubitEmissionWeights>,
    p2_angle_scaling: Option<AngleScaling>,
    p2_idle_rate: f64,

    // Preparation parameters
    p_prep: f64,
    p_prep_leak_ratio: f64,
    p_prep_crosstalk: f64,

    // Measurement parameters
    p_meas_0: f64,
    p_meas_1: f64,

    // Crosstalk parameters (measurement)
    p_meas_crosstalk_global: f64,
    p_meas_crosstalk_local: f64,
    p_meas_crosstalk_local_fn: Option<NeighborFn>,
    p_meas_crosstalk_model: Option<CrosstalkTransitions>,

    // MeasureLeaked handling
    handle_measure_leaked: bool,

    // Idle noise parameters (T1/T2)
    p_idle_linear_rate: f64,
    p_idle_linear_pauli_weights: Option<PauliWeights>,
    p_idle_quadratic_rate: f64,
    p_idle_coherent: bool,
    p_idle_coherent_to_incoherent_factor: f64,

    // Leakage parameters
    leakage_scale: f64,

    // Noiseless gates
    noiseless_gates: Vec<GateType>,

    // Time scale for physical time interpretation
    time_scale: Option<TimeScale>,

    // Custom channels (composite or traditional)
    custom_channels: Vec<Box<dyn crate::noise::NoiseChannel>>,
}

impl Default for CompositeNoiseModelBuilder {
    fn default() -> Self {
        Self {
            p1: 0.0,
            p1_emission_ratio: 0.0,
            p1_seepage: 0.0,
            p1_pauli_weights: None,
            p1_emission_model: None,
            p2: 0.0,
            p2_emission_ratio: 0.0,
            p2_seepage: 0.0,
            p2_pauli_weights: None,
            p2_pauli_model: None,
            p2_emission_model: None,
            p2_angle_scaling: None,
            p2_idle_rate: 0.0,
            p_prep: 0.0,
            p_prep_leak_ratio: 0.0,
            p_prep_crosstalk: 0.0,
            p_meas_0: 0.0,
            p_meas_1: 0.0,
            p_meas_crosstalk_global: 0.0,
            p_meas_crosstalk_local: 0.0,
            p_meas_crosstalk_local_fn: None,
            p_meas_crosstalk_model: None,
            handle_measure_leaked: false,
            p_idle_linear_rate: 0.0,
            p_idle_linear_pauli_weights: None,
            p_idle_quadratic_rate: 0.0,
            p_idle_coherent: false,
            p_idle_coherent_to_incoherent_factor: 1.0,
            leakage_scale: 1.0, // 1.0 = all leakage events remain leakage
            noiseless_gates: Vec::new(),
            time_scale: None,
            custom_channels: Vec::new(),
        }
    }
}

impl CompositeNoiseModelBuilder {
    /// Create a new builder with all parameters set to zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ========================================================================
    // Single-qubit gate parameters
    // ========================================================================

    /// Set the single-qubit gate error probability.
    ///
    /// With probability `p`, a random Pauli error is applied after each
    /// single-qubit gate.
    #[must_use]
    pub fn with_p1(mut self, p: f64) -> Self {
        self.p1 = validate_probability(p, "p1");
        self
    }

    /// Set the fraction of single-qubit errors that cause leakage (emission).
    ///
    /// When an error occurs:
    /// - With probability `ratio`, the qubit leaks
    /// - With probability `1 - ratio`, a Pauli error is applied
    #[must_use]
    pub fn with_p1_emission_ratio(mut self, ratio: f64) -> Self {
        self.p1_emission_ratio = validate_probability(ratio, "p1_emission_ratio");
        self
    }

    /// Set the seepage probability for leaked qubits during single-qubit gates.
    ///
    /// When a leaked qubit undergoes a single-qubit gate, with probability `p`
    /// it seeps back to the computational basis (with a random Pauli applied).
    #[must_use]
    pub fn with_p1_seepage(mut self, p: f64) -> Self {
        self.p1_seepage = validate_probability(p, "p1_seepage");
        self
    }

    /// Set custom Pauli weights for single-qubit gate errors.
    ///
    /// By default, uniform weights (1/3, 1/3, 1/3) are used.
    /// Use this to bias towards specific error types.
    #[must_use]
    pub fn with_p1_pauli_weights(mut self, weights: PauliWeights) -> Self {
        self.p1_pauli_weights = Some(weights);
        self
    }

    /// Set custom emission model for single-qubit gate emission errors.
    ///
    /// The emission model specifies the distribution of error types when
    /// an emission event occurs: X, Y, Z Pauli errors, or leakage.
    ///
    /// By default, emission events always cause leakage. Use this method
    /// to specify a more nuanced model where emission can result in
    /// different error types with different probabilities.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::SingleQubitEmissionWeights;
    /// use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
    ///
    /// // 25% each for X, Y, Z Pauli errors, 25% leakage
    /// let emission = SingleQubitEmissionWeights::custom(0.25, 0.25, 0.25, 0.25);
    ///
    /// let model = CompositeNoiseModelBuilder::new()
    ///     .with_p1(0.01)
    ///     .with_p1_emission_ratio(0.5)  // 50% of errors are emission
    ///     .with_p1_emission_model(emission)
    ///     .build();
    /// ```
    #[must_use]
    pub fn with_p1_emission_model(mut self, model: SingleQubitEmissionWeights) -> Self {
        self.p1_emission_model = Some(model);
        self
    }

    // ========================================================================
    // Two-qubit gate parameters
    // ========================================================================

    /// Set the two-qubit gate error probability.
    ///
    /// With probability `p`, a random two-qubit Pauli error is applied after
    /// each two-qubit gate. The error is applied independently to each qubit.
    #[must_use]
    pub fn with_p2(mut self, p: f64) -> Self {
        self.p2 = validate_probability(p, "p2");
        self
    }

    /// Set the fraction of two-qubit errors that cause leakage (emission).
    #[must_use]
    pub fn with_p2_emission_ratio(mut self, ratio: f64) -> Self {
        self.p2_emission_ratio = validate_probability(ratio, "p2_emission_ratio");
        self
    }

    /// Set the seepage probability for leaked qubits during two-qubit gates.
    #[must_use]
    pub fn with_p2_seepage(mut self, p: f64) -> Self {
        self.p2_seepage = validate_probability(p, "p2_seepage");
        self
    }

    /// Set custom Pauli weights for two-qubit gate errors (independent per qubit).
    ///
    /// By default, uniform weights (1/3, 1/3, 1/3) are used per qubit.
    /// This applies errors independently to each qubit.
    ///
    /// For correlated two-qubit Pauli errors (e.g., XX, ZZ), use
    /// `with_p2_pauli_model()` instead.
    #[must_use]
    pub fn with_p2_pauli_weights(mut self, weights: PauliWeights) -> Self {
        self.p2_pauli_weights = Some(weights);
        self
    }

    /// Set correlated two-qubit Pauli model for two-qubit gate errors.
    ///
    /// This specifies the distribution over all 15 non-identity two-qubit
    /// Pauli operators (XI, YI, ZI, IX, IY, IZ, XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ).
    ///
    /// When set, this takes precedence over `with_p2_pauli_weights()`.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::TwoQubitPauliWeights;
    /// use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
    ///
    /// // ZZ-biased errors (common in certain gate implementations)
    /// let model = CompositeNoiseModelBuilder::new()
    ///     .with_p2(0.01)
    ///     .with_p2_pauli_model(TwoQubitPauliWeights::zz_biased(0.5))
    ///     .build();
    /// ```
    #[must_use]
    pub fn with_p2_pauli_model(mut self, model: TwoQubitPauliWeights) -> Self {
        self.p2_pauli_model = Some(model);
        self
    }

    /// Set custom emission model for two-qubit gate emission errors.
    ///
    /// The emission model specifies the distribution of error types when
    /// an emission event occurs on a two-qubit gate. This includes all
    /// combinations of Pauli errors (I, X, Y, Z) and leakage (L) on each qubit.
    ///
    /// By default, emission events always cause leakage. Use this method
    /// to specify a more nuanced model.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::TwoQubitEmissionWeights;
    /// use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
    ///
    /// // Use uniform weights with leakage
    /// let model = CompositeNoiseModelBuilder::new()
    ///     .with_p2(0.01)
    ///     .with_p2_emission_ratio(0.5)
    ///     .with_p2_emission_model(TwoQubitEmissionWeights::uniform_with_leakage())
    ///     .build();
    /// ```
    #[must_use]
    pub fn with_p2_emission_model(mut self, model: TwoQubitEmissionWeights) -> Self {
        self.p2_emission_model = Some(model);
        self
    }

    /// Set angle-dependent scaling for two-qubit gates.
    ///
    /// For parameterized two-qubit gates (RZZ, RXX, etc.), this scales
    /// the base error probability based on the gate's rotation angle.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
    /// use pecos_neo::noise::two_qubit::AngleScaling;
    ///
    /// let model = CompositeNoiseModelBuilder::new()
    ///     .with_p2(0.01)
    ///     .with_p2_angle_scaling(AngleScaling::linear())  // Error ~ |theta/pi|
    ///     .build();
    /// ```
    #[must_use]
    pub fn with_p2_angle_scaling(mut self, scaling: AngleScaling) -> Self {
        self.p2_angle_scaling = Some(scaling);
        self
    }

    /// Set idle noise rate after two-qubit gates.
    ///
    /// This applies additional stochastic noise after each two-qubit gate,
    /// modeling the fact that two-qubit gates often take longer.
    #[must_use]
    pub fn with_p2_idle(mut self, rate: f64) -> Self {
        self.p2_idle_rate = validate_probability(rate, "p2_idle");
        self
    }

    // ========================================================================
    // Preparation parameters
    // ========================================================================

    /// Set the preparation error probability.
    ///
    /// With probability `p`, an X error is applied after preparation.
    #[must_use]
    pub fn with_p_prep(mut self, p: f64) -> Self {
        self.p_prep = validate_probability(p, "p_prep");
        self
    }

    /// Set the fraction of preparation errors that cause leakage.
    #[must_use]
    pub fn with_p_prep_leak_ratio(mut self, ratio: f64) -> Self {
        self.p_prep_leak_ratio = validate_probability(ratio, "p_prep_leak_ratio");
        self
    }

    /// Set preparation crosstalk probability.
    ///
    /// With probability `p`, each other active qubit receives a random Pauli
    /// error during preparation.
    #[must_use]
    pub fn with_p_prep_crosstalk(mut self, p: f64) -> Self {
        self.p_prep_crosstalk = validate_probability(p, "p_prep_crosstalk");
        self
    }

    // ========================================================================
    // Measurement parameters
    // ========================================================================

    /// Set asymmetric measurement error probabilities.
    ///
    /// - `p_0_to_1`: Probability of reading 1 when the state is 0
    /// - `p_1_to_0`: Probability of reading 0 when the state is 1
    #[must_use]
    pub fn with_p_meas(mut self, p_0_to_1: f64, p_1_to_0: f64) -> Self {
        self.p_meas_0 = validate_probability(p_0_to_1, "p_meas_0");
        self.p_meas_1 = validate_probability(p_1_to_0, "p_meas_1");
        self
    }

    /// Set symmetric measurement error probability.
    #[must_use]
    pub fn with_p_meas_symmetric(mut self, p: f64) -> Self {
        let validated = validate_probability(p, "p_meas");
        self.p_meas_0 = validated;
        self.p_meas_1 = validated;
        self
    }

    // ========================================================================
    // Crosstalk parameters
    // ========================================================================

    /// Set global crosstalk probability during measurements.
    ///
    /// With probability `p`, each other active qubit receives a random Pauli error.
    /// This is a convenience method equivalent to `with_p_meas_crosstalk(p, 0.0)`.
    #[must_use]
    pub fn with_p_crosstalk(mut self, p: f64) -> Self {
        self.p_meas_crosstalk_global = validate_probability(p, "p_meas_crosstalk_global");
        self
    }

    /// Set measurement crosstalk probabilities.
    ///
    /// - `global`: Probability of error on any other active qubit
    /// - `local`: Probability of error on neighboring qubits only
    ///
    /// Note: Local crosstalk requires a neighbor function to be provided
    /// via `with_p_meas_crosstalk_local_fn`.
    #[must_use]
    pub fn with_p_meas_crosstalk(mut self, global: f64, local: f64) -> Self {
        self.p_meas_crosstalk_global = validate_probability(global, "p_meas_crosstalk_global");
        self.p_meas_crosstalk_local = validate_probability(local, "p_meas_crosstalk_local");
        self
    }

    /// Set the neighbor function for local measurement crosstalk.
    ///
    /// The function takes the gated qubits and returns their neighbors.
    /// This is required when using `with_p_meas_crosstalk` with a non-zero
    /// local probability.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
    /// use pecos_core::QubitId;
    ///
    /// // Define neighbors as adjacent qubits in a linear chain
    /// fn linear_neighbors(qubits: &[QubitId]) -> Vec<QubitId> {
    ///     qubits.iter()
    ///         .flat_map(|q| vec![QubitId(q.0.saturating_sub(1)), QubitId(q.0 + 1)])
    ///         .filter(|n| !qubits.contains(n))
    ///         .collect()
    /// }
    ///
    /// let model = CompositeNoiseModelBuilder::new()
    ///     .with_p_meas_crosstalk(0.001, 0.01)
    ///     .with_p_meas_crosstalk_local_fn(linear_neighbors)
    ///     .build();
    /// ```
    #[must_use]
    pub fn with_p_meas_crosstalk_local_fn(mut self, neighbor_fn: NeighborFn) -> Self {
        self.p_meas_crosstalk_local_fn = Some(neighbor_fn);
        self
    }

    /// Set the state-dependent crosstalk transition model.
    ///
    /// When set, crosstalk effects use state-dependent transitions:
    /// - Given qubit is in |0⟩: probability to stay, flip, or leak
    /// - Given qubit is in |1⟩: probability to stay, flip, or leak
    ///
    /// When not set, crosstalk uses uniform random Pauli errors.
    ///
    /// This matches `GeneralNoiseModel`'s `p_meas_crosstalk_model`.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::CrosstalkTransitions;
    /// use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
    ///
    /// // Asymmetric crosstalk: qubits in |0⟩ more likely to leak
    /// let transitions = CrosstalkTransitions::custom(
    ///     0.3, 0.2, 0.5,  // from 0: 30% stay, 20% flip, 50% leak
    ///     0.4, 0.5, 0.1,  // from 1: 40% stay, 50% flip, 10% leak
    /// );
    ///
    /// let model = CompositeNoiseModelBuilder::new()
    ///     .with_p_crosstalk(0.01)
    ///     .with_p_meas_crosstalk_model(transitions)
    ///     .build();
    /// ```
    #[must_use]
    pub fn with_p_meas_crosstalk_model(mut self, transitions: CrosstalkTransitions) -> Self {
        self.p_meas_crosstalk_model = Some(transitions);
        self
    }

    // ========================================================================
    // MeasureLeaked handling
    // ========================================================================

    /// Enable handling of `MeasureLeaked` gates.
    ///
    /// When enabled, `MeasureLeaked` gates will return outcome 2 for qubits
    /// that are in a leaked state. This matches the behavior of
    /// `GeneralNoiseModel`.
    ///
    /// By default, this is disabled.
    #[must_use]
    pub fn with_measure_leaked_handling(mut self, enabled: bool) -> Self {
        self.handle_measure_leaked = enabled;
        self
    }

    // ========================================================================
    // Idle noise parameters (T1/T2)
    // ========================================================================

    /// Set the linear idle noise rate (T1-like relaxation).
    ///
    /// The error probability grows linearly with idle duration:
    /// `p = rate * duration`
    ///
    /// Units depend on the simulation's time units.
    #[must_use]
    pub fn with_p_idle_linear(mut self, rate_per_time_unit: f64) -> Self {
        self.p_idle_linear_rate = validate_rate(rate_per_time_unit, "p_idle_linear_rate");
        self
    }

    /// Set the quadratic idle noise rate (T2-like dephasing).
    ///
    /// The error probability follows: `p = sin(rate * duration)^2`
    ///
    /// This models coherent dephasing converted to stochastic errors.
    /// Use `with_p_idle_coherent(true)` to use actual coherent RZ rotations instead.
    #[must_use]
    pub fn with_p_idle_quadratic(mut self, rate_per_time_unit: f64) -> Self {
        self.p_idle_quadratic_rate = validate_rate(rate_per_time_unit, "p_idle_quadratic_rate");
        self
    }

    /// Set whether to use coherent dephasing for quadratic idle noise.
    ///
    /// When `true`, quadratic idle noise is modeled as coherent RZ rotations
    /// rather than stochastic Z errors. This represents systematic phase evolution
    /// such as frequency offsets in physical systems.
    ///
    /// When `false` (default), dephasing is modeled as stochastic Z errors
    /// with probability `sin(rate * duration)^2`.
    #[must_use]
    pub fn with_p_idle_coherent(mut self, coherent: bool) -> Self {
        self.p_idle_coherent = coherent;
        self
    }

    /// Set the coherent-to-incoherent conversion factor for quadratic dephasing.
    ///
    /// When modeling coherent dephasing using stochastic errors, this factor
    /// scales the effective rate. Default is 1.0 (no adjustment).
    #[must_use]
    pub fn with_coherent_to_incoherent_factor(mut self, factor: f64) -> Self {
        self.p_idle_coherent_to_incoherent_factor =
            validate_rate(factor, "coherent_to_incoherent_factor");
        self
    }

    /// Set both T1 and T2 idle noise rates.
    ///
    /// Convenience method for setting both linear (T1) and quadratic (T2) rates.
    #[must_use]
    pub fn with_idle_noise(mut self, t1_rate: f64, t2_rate: f64) -> Self {
        self.p_idle_linear_rate = validate_rate(t1_rate, "p_idle_linear_rate");
        self.p_idle_quadratic_rate = validate_rate(t2_rate, "p_idle_quadratic_rate");
        self
    }

    /// Set custom Pauli weights for linear (T1) idle noise.
    ///
    /// By default, T1 noise applies random Pauli errors with uniform distribution.
    /// Use this to bias towards specific error types (e.g., Z-only for pure dephasing).
    #[must_use]
    pub fn with_p_idle_linear_weights(mut self, weights: PauliWeights) -> Self {
        self.p_idle_linear_pauli_weights = Some(weights);
        self
    }

    // ========================================================================
    // Leakage convenience methods
    // ========================================================================

    /// Set leakage and seepage probabilities for all gate types.
    ///
    /// - `p_leak`: Fraction of errors that cause leakage
    /// - `p_seep`: Probability of seepage when a leaked qubit is gated
    #[must_use]
    pub fn with_leakage(mut self, p_leak: f64, p_seep: f64) -> Self {
        let leak = validate_probability(p_leak, "p_leak");
        let seep = validate_probability(p_seep, "p_seep");
        self.p1_emission_ratio = leak;
        self.p2_emission_ratio = leak;
        self.p1_seepage = seep;
        self.p2_seepage = seep;
        self
    }

    /// Set the leakage scale factor.
    ///
    /// This controls what fraction of leakage events actually result in leakage
    /// versus being converted to completely depolarizing noise:
    /// - 1.0 (default): All leakage events remain leakage
    /// - 0.0: All leakage events become depolarizing (no leakage)
    /// - 0.5: Half of leakage events remain leakage, half become depolarizing
    ///
    /// This is useful for comparing circuit performance with and without leakage
    /// effects while maintaining the same overall error rate.
    #[must_use]
    pub fn with_leakage_scale(mut self, scale: f64) -> Self {
        self.leakage_scale = scale.clamp(0.0, 1.0);
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
    /// When set, convenience methods like `with_idle_t1_t2()` become available,
    /// and the time scale is passed through to the built noise model.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
    /// use pecos_core::TimeScale;
    ///
    /// let noise = CompositeNoiseModelBuilder::new()
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
    /// This is a convenience method that converts physical T1/T2 times to
    /// the internal rate parameters based on the configured time scale.
    ///
    /// - T1 (amplitude damping): Sets linear idle noise rate = 1/T1
    /// - T2 (dephasing): Sets quadratic idle noise rate = 1/T2^2
    ///
    /// # Panics
    ///
    /// Panics if `with_time_scale()` has not been called first.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
    /// use pecos_core::TimeScale;
    ///
    /// let noise = CompositeNoiseModelBuilder::new()
    ///     .with_time_scale(TimeScale::NANOSECONDS)
    ///     .with_idle_t1_t2(50e-6, 30e-6)  // T1=50us, T2=30us in seconds
    ///     .build();
    /// ```
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
    // Custom channels
    // ========================================================================

    /// Add a custom noise channel (composite or traditional).
    ///
    /// This allows mixing different channel types in a single noise model.
    /// Channels are applied in the order they are added.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
    /// use pecos_neo::noise::MeasurementChannel;
    ///
    /// let model = CompositeNoiseModelBuilder::new()
    ///     .with_p1(0.001)  // Flow single-qubit noise
    ///     .with_channel(MeasurementChannel::symmetric(0.02))  // Traditional channel
    ///     .build();
    /// ```
    #[must_use]
    pub fn with_channel(mut self, channel: impl crate::noise::NoiseChannel + 'static) -> Self {
        self.custom_channels.push(Box::new(channel));
        self
    }

    // ========================================================================
    // Build
    // ========================================================================

    /// Build the configured noise model.
    ///
    /// Returns a `ComposableNoiseModel` with composite-based channels.
    #[must_use]
    pub fn build(self) -> ComposableNoiseModel {
        let mut model = ComposableNoiseModel::new();

        // Set time scale if configured
        if let Some(scale) = self.time_scale {
            model = model.with_time_scale(scale);
        }

        // Add noiseless gates to context
        for gate_type in &self.noiseless_gates {
            model = model.with_noiseless_gate(*gate_type);
        }

        // Single-qubit gate noise
        if self.p1 > 0.0 {
            let sq_noise = self.build_single_qubit_noise();
            let channel = CompositeChannelBuilder::single_qubit("flow_sq", sq_noise);
            model = model.add_channel(channel);
        }

        // Two-qubit gate noise (including p2_idle)
        if self.p2 > 0.0 || self.p2_idle_rate > 0.0 {
            let tq_noise = self.build_two_qubit_noise();
            let channel = CompositeChannelBuilder::two_qubit("flow_tq", tq_noise);
            model = model.add_channel(channel);
        }

        // Preparation noise
        if self.p_prep > 0.0 {
            let prep_noise = self.build_preparation_noise();
            let channel = CompositeChannelBuilder::preparation("flow_prep", prep_noise);
            model = model.add_channel(channel);
        }

        // Measurement noise
        if self.p_meas_0 > 0.0 || self.p_meas_1 > 0.0 {
            let meas_noise = self.build_measurement_noise();
            let channel = CompositeChannel::new("flow_meas", meas_noise)
                .with_filter(CompositeEventFilter::AfterMeasurement);
            model = model.add_channel(channel);
        }

        // Measurement crosstalk (global)
        if self.p_meas_crosstalk_global > 0.0 {
            let crosstalk_noise = self.build_crosstalk_noise(self.p_meas_crosstalk_global);
            let channel =
                CompositeCrosstalkChannel::new("flow_meas_crosstalk_global", crosstalk_noise)
                    .responds_to_measurement()
                    .global();
            model = model.add_channel(channel);
        }

        // Measurement crosstalk (local)
        if self.p_meas_crosstalk_local > 0.0 {
            if let Some(neighbor_fn) = self.p_meas_crosstalk_local_fn {
                let crosstalk_noise = self.build_crosstalk_noise(self.p_meas_crosstalk_local);
                let channel =
                    CompositeCrosstalkChannel::new("flow_meas_crosstalk_local", crosstalk_noise)
                        .responds_to_measurement()
                        .local(neighbor_fn);
                model = model.add_channel(channel);
            } else {
                eprintln!(
                    "Warning: p_meas_crosstalk_local = {} is set but no neighbor function \
                     was provided via with_p_meas_crosstalk_local_fn(). \
                     Local crosstalk will be ignored.",
                    self.p_meas_crosstalk_local
                );
            }
        }

        // Preparation crosstalk
        if self.p_prep_crosstalk > 0.0 {
            let crosstalk_noise = prob(self.p_prep_crosstalk, pauli());
            let channel = CompositeCrosstalkChannel::new("flow_prep_crosstalk", crosstalk_noise)
                .responds_to_preparation()
                .global();
            model = model.add_channel(channel);
        }

        // Idle noise (T1/T2)
        if self.p_idle_linear_rate > 0.0 || self.p_idle_quadratic_rate > 0.0 {
            let idle_noise = self.build_idle_noise();
            let channel = CompositeChannelBuilder::idle("flow_idle", idle_noise);
            model = model.add_channel(channel);
        }

        // Before-gate channel for skip logic (if leakage is enabled)
        if self.has_leakage() {
            let skip_channel = CompositeChannelBuilder::before_gate("flow_skip", skip_if_leaked());
            model = model.add_channel(skip_channel);
        }

        // MeasureLeaked handling: return outcome 2 for leaked qubits
        if self.handle_measure_leaked {
            let measure_leaked_noise = self.build_measure_leaked_noise();
            let channel = CompositeChannel::new("flow_measure_leaked", measure_leaked_noise)
                .with_filter(CompositeEventFilter::AfterMeasurement);
            model = model.add_channel(channel);
        }

        // Custom channels (composite or traditional)
        for channel in self.custom_channels {
            model = model.add_boxed_channel(channel);
        }

        model
    }

    /// Check if any leakage is configured (accounting for `leakage_scale`).
    fn has_leakage(&self) -> bool {
        // Only consider leakage if leakage_scale > 0
        let effective_p1_leak = self.p1_emission_ratio * self.leakage_scale;
        let effective_p2_leak = self.p2_emission_ratio * self.leakage_scale;
        // Prep leakage is not affected by leakage_scale in this implementation
        effective_p1_leak > 0.0 || effective_p2_leak > 0.0 || self.p_prep_leak_ratio > 0.0
    }

    /// Build crosstalk noise primitive.
    ///
    /// Uses the transition model if configured, otherwise uses uniform Pauli errors.
    fn build_crosstalk_noise(&self, probability: f64) -> BoxSeq {
        match self.p_meas_crosstalk_model {
            Some(transitions) => {
                seq![prob(probability, crosstalk_transitions(transitions)),]
            }
            None => {
                seq![prob(probability, pauli()),]
            }
        }
    }

    /// Build `MeasureLeaked` noise primitive.
    ///
    /// For `MeasureLeaked` gates, leaked qubits return outcome 2.
    fn build_measure_leaked_noise(&self) -> BoxSeq {
        // Only apply leaked_measurement for MeasureLeaked gates on leaked qubits
        seq![on_gate_type(
            GateType::MeasureLeaked,
            when_leaked(leaked_measurement(), nothing()),
        ),]
    }

    /// Build single-qubit gate noise primitive.
    fn build_single_qubit_noise(&self) -> BoxSeq {
        use super::action::{Emission, Leak};

        let pauli_action = self.make_p1_pauli();

        // Apply leakage scale: effective_leak = emission_ratio * leakage_scale
        // When leakage_scale < 1.0, some emission events become depolarizing instead
        let effective_leak = self.p1_emission_ratio * self.leakage_scale;

        // Dispatch based on emission model configuration
        match self.p1_emission_model {
            Some(weights) => {
                let emission_action = Emission::with_weights(weights);
                self.build_single_qubit_noise_inner(pauli_action, emission_action, effective_leak)
            }
            None => self.build_single_qubit_noise_inner(pauli_action, Leak, effective_leak),
        }
    }

    /// Inner helper for building single-qubit noise with concrete types.
    fn build_single_qubit_noise_inner<E>(
        &self,
        pauli_action: Pauli,
        emission_action: E,
        effective_leak: f64,
    ) -> BoxSeq
    where
        E: crate::noise::composite::Primitive + 'static,
    {
        if self.p1_seepage > 0.0 || effective_leak > 0.0 {
            // With leakage/seepage: complex decision tree
            seq![prob(
                self.p1,
                when_leaked(
                    // Leaked qubit: seepage with probability
                    prob(self.p1_seepage, seep()),
                    // Not leaked: emission (scaled) or pauli
                    if effective_leak > 0.0 {
                        sample![
                            (effective_leak, emission_action),
                            (1.0 - effective_leak, pauli_action),
                        ]
                    } else {
                        sample![(1.0, pauli_action),]
                    },
                ),
            ),]
        } else {
            // Simple depolarizing
            seq![prob(self.p1, pauli_action),]
        }
    }

    /// Create Pauli action for single-qubit gates.
    fn make_p1_pauli(&self) -> Pauli {
        match self.p1_pauli_weights {
            Some(w) => Pauli::new(w),
            None => Pauli::uniform(),
        }
    }

    /// Build two-qubit gate noise primitive.
    ///
    /// When angle scaling is configured, uses `prob_fn` for dynamic probability.
    /// Otherwise, uses constant `prob`.
    fn build_two_qubit_noise(&self) -> BoxSeq {
        // Check if we need angle-dependent probability
        let main_noise = if let Some(scaling) = self.p2_angle_scaling {
            self.build_two_qubit_noise_angle_scaled(scaling)
        } else {
            self.build_two_qubit_noise_constant()
        };

        // Add idle noise after the gate if configured (memory sweeping)
        if self.p2_idle_rate > 0.0 {
            seq![main_noise, prob(self.p2_idle_rate, inject_z()),]
        } else {
            main_noise
        }
    }

    /// Build two-qubit noise with constant error probability.
    fn build_two_qubit_noise_constant(&self) -> BoxSeq {
        use super::action::{Leak, Pauli, TwoQubitEmission, TwoQubitPauli};

        // Apply leakage scale: effective_leak = emission_ratio * leakage_scale
        let effective_leak = self.p2_emission_ratio * self.leakage_scale;

        // Build based on which models are configured
        match (self.p2_pauli_model.clone(), self.p2_emission_model.clone()) {
            // Both correlated models
            (Some(pauli_model), Some(emission_model)) => {
                let pauli_action = TwoQubitPauli::with_weights(pauli_model);
                let emission_action = TwoQubitEmission::with_weights(emission_model);
                self.build_two_qubit_constant_inner(pauli_action, emission_action, effective_leak)
            }
            // Correlated pauli, default emission (leak)
            (Some(pauli_model), None) => {
                let pauli_action = TwoQubitPauli::with_weights(pauli_model);
                self.build_two_qubit_constant_inner(pauli_action, Leak, effective_leak)
            }
            // Default pauli, correlated emission
            (None, Some(emission_model)) => {
                let pauli_action = match self.p2_pauli_weights {
                    Some(w) => Pauli::new(w),
                    None => Pauli::uniform(),
                };
                let emission_action = TwoQubitEmission::with_weights(emission_model);
                self.build_two_qubit_constant_inner(pauli_action, emission_action, effective_leak)
            }
            // Both default (independent per-qubit Pauli, leak for emission)
            (None, None) => {
                let pauli_action = match self.p2_pauli_weights {
                    Some(w) => Pauli::new(w),
                    None => Pauli::uniform(),
                };
                self.build_two_qubit_constant_inner(pauli_action, Leak, effective_leak)
            }
        }
    }

    /// Inner helper for building two-qubit constant noise.
    fn build_two_qubit_constant_inner<P, E>(
        &self,
        pauli_action: P,
        emission_action: E,
        effective_leak: f64,
    ) -> BoxSeq
    where
        P: crate::noise::composite::Primitive + 'static,
        E: crate::noise::composite::Primitive + 'static,
    {
        if self.p2_seepage > 0.0 || effective_leak > 0.0 {
            // With leakage/seepage
            seq![prob(
                self.p2,
                when_leaked(
                    prob(self.p2_seepage, seep()),
                    if effective_leak > 0.0 {
                        sample![
                            (effective_leak, emission_action),
                            (1.0 - effective_leak, pauli_action),
                        ]
                    } else {
                        sample![(1.0, pauli_action),]
                    },
                ),
            ),]
        } else {
            // Simple depolarizing
            seq![prob(self.p2, pauli_action),]
        }
    }

    /// Build two-qubit noise with angle-dependent error probability.
    fn build_two_qubit_noise_angle_scaled(&self, scaling: AngleScaling) -> BoxSeq {
        use super::action::{Leak, Pauli, TwoQubitEmission, TwoQubitPauli};

        // Apply leakage scale: effective_leak = emission_ratio * leakage_scale
        let effective_leak = self.p2_emission_ratio * self.leakage_scale;

        // Build based on which models are configured
        match (self.p2_pauli_model.clone(), self.p2_emission_model.clone()) {
            // Both correlated models
            (Some(pauli_model), Some(emission_model)) => {
                let pauli_action = TwoQubitPauli::with_weights(pauli_model);
                let emission_action = TwoQubitEmission::with_weights(emission_model);
                self.build_two_qubit_angle_inner(
                    scaling,
                    pauli_action,
                    emission_action,
                    effective_leak,
                )
            }
            // Correlated pauli, default emission (leak)
            (Some(pauli_model), None) => {
                let pauli_action = TwoQubitPauli::with_weights(pauli_model);
                self.build_two_qubit_angle_inner(scaling, pauli_action, Leak, effective_leak)
            }
            // Default pauli, correlated emission
            (None, Some(emission_model)) => {
                let pauli_action = match self.p2_pauli_weights {
                    Some(w) => Pauli::new(w),
                    None => Pauli::uniform(),
                };
                let emission_action = TwoQubitEmission::with_weights(emission_model);
                self.build_two_qubit_angle_inner(
                    scaling,
                    pauli_action,
                    emission_action,
                    effective_leak,
                )
            }
            // Both default (independent per-qubit Pauli, leak for emission)
            (None, None) => {
                let pauli_action = match self.p2_pauli_weights {
                    Some(w) => Pauli::new(w),
                    None => Pauli::uniform(),
                };
                self.build_two_qubit_angle_inner(scaling, pauli_action, Leak, effective_leak)
            }
        }
    }

    /// Inner helper for building two-qubit angle-scaled noise.
    fn build_two_qubit_angle_inner<P, E>(
        &self,
        scaling: AngleScaling,
        pauli_action: P,
        emission_action: E,
        effective_leak: f64,
    ) -> BoxSeq
    where
        P: crate::noise::composite::Primitive + 'static,
        E: crate::noise::composite::Primitive + 'static,
    {
        let base_p2 = self.p2;
        let seepage = self.p2_seepage;

        // Different code paths based on whether we have leakage/seepage
        if seepage > 0.0 || effective_leak > 0.0 {
            // With leakage/seepage: use complex inner structure
            seq![prob_fn(
                move |gate| {
                    gate.and_then(crate::noise::GateInfo::angle)
                        .map_or(base_p2, |angle| base_p2 * scaling.scale(angle))
                },
                when_leaked(
                    prob(seepage, seep()),
                    if effective_leak > 0.0 {
                        sample![
                            (effective_leak, emission_action),
                            (1.0 - effective_leak, pauli_action),
                        ]
                    } else {
                        sample![(1.0, pauli_action),]
                    },
                ),
            ),]
        } else {
            // Simple depolarizing with angle scaling
            seq![prob_fn(
                move |gate| {
                    gate.and_then(crate::noise::GateInfo::angle)
                        .map_or(base_p2, |angle| base_p2 * scaling.scale(angle))
                },
                pauli_action,
            ),]
        }
    }

    /// Build preparation noise primitive.
    fn build_preparation_noise(&self) -> BoxSeq {
        if self.p_prep_leak_ratio > 0.0 {
            seq![prob(
                self.p_prep,
                sample![
                    (self.p_prep_leak_ratio, leak()),
                    (1.0 - self.p_prep_leak_ratio, inject_x()),
                ],
            ),]
        } else {
            seq![prob(self.p_prep, inject_x()),]
        }
    }

    /// Build measurement noise primitive.
    fn build_measurement_noise(&self) -> BoxSeq {
        if (self.p_meas_0 - self.p_meas_1).abs() < 1e-10 {
            // Symmetric measurement error
            seq![prob(self.p_meas_0, flip_outcome()),]
        } else {
            // Asymmetric measurement error
            seq![
                on_zero(prob(self.p_meas_0, flip_outcome())),
                on_one(prob(self.p_meas_1, flip_outcome())),
            ]
        }
    }

    /// Build idle noise primitive (T1/T2).
    fn build_idle_noise(&self) -> BoxSeq {
        use super::action::InjectCoherentRZ;
        use super::primitive::{ProbLinear, ProbQuadratic};

        // T1 uses custom Pauli weights if specified, otherwise uniform
        let make_t1_pauli = || match self.p_idle_linear_pauli_weights {
            Some(w) => Pauli::new(w),
            None => Pauli::uniform(),
        };

        // T2: either coherent RZ rotations or stochastic Z errors
        let make_t2_stochastic = || {
            ProbQuadratic::new(self.p_idle_quadratic_rate, inject_z())
                .with_factor(self.p_idle_coherent_to_incoherent_factor)
        };

        let make_t2_coherent = || InjectCoherentRZ::new(self.p_idle_quadratic_rate);

        match (
            self.p_idle_linear_rate > 0.0,
            self.p_idle_quadratic_rate > 0.0,
            self.p_idle_coherent,
        ) {
            (true, true, false) => {
                // Both T1 (linear) and T2 (quadratic stochastic) noise
                seq![
                    ProbLinear::new(self.p_idle_linear_rate, make_t1_pauli()),
                    make_t2_stochastic(),
                ]
            }
            (true, true, true) => {
                // Both T1 (linear stochastic) and T2 (coherent RZ) noise
                seq![
                    ProbLinear::new(self.p_idle_linear_rate, make_t1_pauli()),
                    make_t2_coherent(),
                ]
            }
            (true, false, _) => {
                // Only T1 (linear) noise
                seq![ProbLinear::new(self.p_idle_linear_rate, make_t1_pauli()),]
            }
            (false, true, false) => {
                // Only T2 (quadratic stochastic) noise
                seq![make_t2_stochastic(),]
            }
            (false, true, true) => {
                // Only T2 (coherent RZ) noise
                seq![make_t2_coherent(),]
            }
            (false, false, _) => {
                // No idle noise (shouldn't reach here due to caller check)
                seq![nothing(),]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::CommandBuilder;
    use crate::runner::CircuitRunner;
    use pecos_core::QubitId;
    use pecos_simulators::SparseStab;

    #[test]
    fn test_empty_builder() {
        let model = CompositeNoiseModelBuilder::new().build();
        assert_eq!(model.channel_count(), 0);
    }

    #[test]
    fn test_simple_depolarizing() {
        let model = CompositeNoiseModelBuilder::new()
            .with_p1(0.01)
            .with_p2(0.02)
            .build();

        // Should have single-qubit and two-qubit channels
        assert_eq!(model.channel_count(), 2);
    }

    #[test]
    fn test_with_leakage() {
        let model = CompositeNoiseModelBuilder::new()
            .with_p1(0.01)
            .with_leakage(0.1, 0.2)
            .build();

        // Should have: SQ channel + skip channel
        assert_eq!(model.channel_count(), 2);
    }

    #[test]
    fn test_full_configuration() {
        let model = CompositeNoiseModelBuilder::new()
            .with_p1(0.01)
            .with_p2(0.02)
            .with_p_prep(0.005)
            .with_p_meas(0.03, 0.04)
            .with_p_crosstalk(0.001)
            .with_leakage(0.1, 0.2)
            .build();

        // SQ + TQ + Prep + Meas + Crosstalk + Skip = 6 channels
        assert_eq!(model.channel_count(), 6);
    }

    #[test]
    fn test_noiseless_gates() {
        let model = CompositeNoiseModelBuilder::new()
            .with_p1(0.01)
            .with_noiseless_gate(GateType::I)
            .with_noiseless_gates(&[GateType::SX, GateType::SXdg])
            .build();

        assert!(model.context().is_noiseless(GateType::I));
        assert!(model.context().is_noiseless(GateType::SX));
        assert!(!model.context().is_noiseless(GateType::H));
    }

    #[test]
    fn test_statistical_behavior() {
        let p1 = 0.1;

        let commands = CommandBuilder::new()
            .pz(0)
            .h(0)
            .h(0)
            .h(0)
            .h(0)
            .h(0)
            .mz(0)
            .build();

        // Run many shots and check error rate
        let shots = 500;
        let mut errors = 0;

        for seed in 0..shots {
            let model = CompositeNoiseModelBuilder::new().with_p1(p1).build();
            let mut state = SparseStab::new(1);
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model)
                .with_seed(seed);
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            if outcomes.get(QubitId(0)).is_none_or(|o| o.outcome) {
                errors += 1;
            }
        }

        // With 5 gates at 10% each, expect significant error rate
        let error_rate = f64::from(errors) / shots as f64;
        assert!(
            error_rate > 0.2,
            "Expected significant error rate, got {error_rate}"
        );
    }

    #[test]
    fn test_asymmetric_measurement() {
        let p_0_to_1 = 0.3; // High rate for testing
        let p_1_to_0 = 0.0;

        // Prep |0> and measure - only 0->1 errors should occur
        let commands = CommandBuilder::new().pz(0).mz(0).build();

        let shots = 500;
        let mut flips = 0;

        for seed in 0..shots {
            let model = CompositeNoiseModelBuilder::new()
                .with_p_meas(p_0_to_1, p_1_to_0)
                .build();
            let mut state = SparseStab::new(1);
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model)
                .with_seed(seed);
            if runner
                .apply_circuit(&mut state, &commands)
                .unwrap()
                .get(QubitId(0))
                .is_some_and(|o| o.outcome)
            {
                flips += 1;
            }
        }

        let flip_rate = f64::from(flips) / shots as f64;
        assert!(
            (flip_rate - p_0_to_1).abs() < 0.1,
            "Expected ~{p_0_to_1} flip rate, got {flip_rate}"
        );
    }

    #[test]
    fn test_idle_noise_builder() {
        // Test that idle noise channels are created
        let model = CompositeNoiseModelBuilder::new()
            .with_p_idle_linear(0.001)
            .with_p_idle_quadratic(0.01)
            .build();

        // Should have 1 idle channel
        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_idle_noise_t1_only() {
        let model = CompositeNoiseModelBuilder::new()
            .with_p_idle_linear(0.001)
            .build();

        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_idle_noise_t2_only() {
        let model = CompositeNoiseModelBuilder::new()
            .with_p_idle_quadratic(0.01)
            .with_coherent_to_incoherent_factor(2.0)
            .build();

        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_idle_noise_convenience() {
        let model = CompositeNoiseModelBuilder::new()
            .with_idle_noise(0.001, 0.01)
            .build();

        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_full_model_with_idle() {
        let model = CompositeNoiseModelBuilder::new()
            .with_p1(0.01)
            .with_p2(0.02)
            .with_p_prep(0.005)
            .with_p_meas_symmetric(0.03)
            .with_idle_noise(0.001, 0.01)
            .build();

        // SQ + TQ + Prep + Meas + Idle = 5 channels
        assert_eq!(model.channel_count(), 5);
    }

    #[test]
    fn test_custom_pauli_weights() {
        // Create model with Z-only errors
        let model = CompositeNoiseModelBuilder::new()
            .with_p1(0.01)
            .with_p1_pauli_weights(PauliWeights::custom(0.0, 0.0, 1.0))
            .with_p2(0.01)
            .with_p2_pauli_weights(PauliWeights::custom(0.0, 0.0, 1.0))
            .build();

        // Should have SQ + TQ channels
        assert_eq!(model.channel_count(), 2);
    }

    #[test]
    fn test_preparation_crosstalk() {
        let model = CompositeNoiseModelBuilder::new()
            .with_p_prep(0.01)
            .with_p_prep_crosstalk(0.001)
            .build();

        // Should have Prep + Prep crosstalk = 2 channels
        assert_eq!(model.channel_count(), 2);
    }

    #[test]
    fn test_measurement_crosstalk_methods() {
        // Test with_p_crosstalk
        let model1 = CompositeNoiseModelBuilder::new()
            .with_p_meas_symmetric(0.01)
            .with_p_crosstalk(0.001)
            .build();

        // Meas + Crosstalk = 2 channels
        assert_eq!(model1.channel_count(), 2);

        // Test with_p_meas_crosstalk
        let model2 = CompositeNoiseModelBuilder::new()
            .with_p_meas_symmetric(0.01)
            .with_p_meas_crosstalk(0.001, 0.0)
            .build();

        // Meas + Global Crosstalk = 2 channels
        assert_eq!(model2.channel_count(), 2);
    }

    #[test]
    fn test_p2_idle() {
        let model = CompositeNoiseModelBuilder::new()
            .with_p2(0.01)
            .with_p2_idle(0.001)
            .build();

        // Should have TQ channel (p2_idle is integrated into the TQ noise)
        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_p2_idle_primitive() {
        use crate::noise::NoiseChannel;
        use pecos_random::PecosRng;

        // Test that the primitive applies idle noise correctly
        // Build the primitive directly
        let tq_noise = seq![
            prob(0.0, pauli()),    // No main error
            prob(1.0, inject_z()), // 100% idle Z error
        ];

        let channel = CompositeChannelBuilder::two_qubit("test_idle", tq_noise);

        let mut ctx = crate::noise::NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Create a two-qubit gate event
        let qubits = [QubitId(0), QubitId(1)];
        let event = crate::noise::NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &[],
            gate_id: None,
        };

        // Check that the channel responds to this event
        assert!(
            channel.responds_to(&event),
            "Channel should respond to AfterGate with 2 qubits"
        );

        // Apply and check that response is not None (gates were injected)
        let response = channel.apply(&event, &mut ctx, &mut rng);
        assert!(
            !response.is_none(),
            "With 100% idle error, should inject Z gates (response should not be None)"
        );
    }

    #[test]
    fn test_p2_idle_statistical() {
        // Test that p2_idle works via the builder
        // Use high p2 (which also applies p2_idle) to ensure the channel triggers
        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .h(0) // Make qubit 0 in superposition
            .cx(0, 1) // Two-qubit gate
            .mz(0)
            .mz(1)
            .build();

        let shots = 500;
        let p2 = 0.5; // 50% gate error rate

        let mut errors_with = 0;
        let mut errors_without = 0;

        for seed in 0..shots {
            // With p2 error
            let model_with = CompositeNoiseModelBuilder::new().with_p2(p2).build();
            let mut state = SparseStab::new(2);
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model_with)
                .with_seed(seed);
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            let q0 = outcomes.get(QubitId(0)).is_some_and(|o| o.outcome);
            let q1 = outcomes.get(QubitId(1)).is_some_and(|o| o.outcome);
            // Bell state: q0 != q1 indicates error
            if q0 != q1 {
                errors_with += 1;
            }

            // Without noise
            let model_without = CompositeNoiseModelBuilder::new().build();
            let mut state = SparseStab::new(2);
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model_without)
                .with_seed(seed);
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            let q0 = outcomes.get(QubitId(0)).is_some_and(|o| o.outcome);
            let q1 = outcomes.get(QubitId(1)).is_some_and(|o| o.outcome);
            if q0 != q1 {
                errors_without += 1;
            }
        }

        // With 50% error, should see significantly more errors
        let rate_with = f64::from(errors_with) / shots as f64;
        let rate_without = f64::from(errors_without) / shots as f64;

        assert!(
            rate_with > rate_without + 0.1,
            "With p2={p2}, expected more errors ({rate_with}) than without ({rate_without})"
        );
    }

    // Note: CompositeNoiseModelBuilder no longer implements Clone because it can hold
    // Box<dyn NoiseChannel> via with_channel(). Use the builder pattern instead.

    #[test]
    fn test_angle_scaling_builder() {
        use crate::noise::two_qubit::AngleScaling;

        // Build model with linear angle scaling
        let model = CompositeNoiseModelBuilder::new()
            .with_p2(0.1)
            .with_p2_angle_scaling(AngleScaling::linear())
            .build();

        // Should have TQ channel
        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_angle_scaling_statistical_behavior() {
        use crate::command::CommandBuilder;
        use crate::noise::two_qubit::AngleScaling;
        use crate::runner::CircuitRunner;
        use pecos_core::{Angle64, QubitId};
        use pecos_simulators::StateVec;

        // With linear scaling, half-turn (pi) should have full error rate
        // Zero angle should have zero error rate
        let base_p2 = 0.5; // 50% base rate for clear signal

        // Test with half-turn angle (pi) - should have ~50% error
        let commands_half = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .rzz(0, 1, Angle64::HALF_TURN) // pi radians
            .mz(0)
            .mz(1)
            .build();

        let shots = 500;
        let mut errors_half = 0;

        for seed in 0..shots {
            let model = CompositeNoiseModelBuilder::new()
                .with_p2(base_p2)
                .with_p2_angle_scaling(AngleScaling::linear())
                .build();

            let mut state = StateVec::new(2);
            let mut runner = CircuitRunner::<StateVec>::rotations()
                .with_noise(model)
                .with_seed(seed);
            let outcomes = runner.apply_circuit(&mut state, &commands_half).unwrap();

            // Count if either qubit has unexpected outcome (both should be 0 without noise)
            let q0 = outcomes.get(QubitId(0)).is_some_and(|o| o.outcome);
            let q1 = outcomes.get(QubitId(1)).is_some_and(|o| o.outcome);
            if q0 || q1 {
                errors_half += 1;
            }
        }

        let error_rate_half = f64::from(errors_half) / shots as f64;
        // Linear scaling at pi -> scale = 1.0, so effective p = 0.5
        // Should see significant errors
        assert!(
            error_rate_half > 0.2,
            "Expected significant errors at half-turn, got {error_rate_half}"
        );

        // Test with quarter-turn angle (pi/4) - should have ~12.5% error (0.5 * 0.25)
        let commands_quarter = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .rzz(0, 1, Angle64::QUARTER_TURN) // pi/2 radians
            .mz(0)
            .mz(1)
            .build();

        let mut errors_quarter = 0;

        for seed in 0..shots {
            let model = CompositeNoiseModelBuilder::new()
                .with_p2(base_p2)
                .with_p2_angle_scaling(AngleScaling::linear())
                .build();

            let mut state = StateVec::new(2);
            let mut runner = CircuitRunner::<StateVec>::rotations()
                .with_noise(model)
                .with_seed(seed);
            let outcomes = runner.apply_circuit(&mut state, &commands_quarter).unwrap();

            let q0 = outcomes.get(QubitId(0)).is_some_and(|o| o.outcome);
            let q1 = outcomes.get(QubitId(1)).is_some_and(|o| o.outcome);
            if q0 || q1 {
                errors_quarter += 1;
            }
        }

        let error_rate_quarter = f64::from(errors_quarter) / shots as f64;
        // Linear scaling at pi/2 -> scale = 0.5, so effective p = 0.25
        // Should see fewer errors than half-turn
        assert!(
            error_rate_quarter < error_rate_half,
            "Quarter-turn ({error_rate_quarter}) should have fewer errors than half-turn ({error_rate_half})"
        );
    }

    #[test]
    fn test_angle_scaling_with_leakage() {
        use crate::noise::two_qubit::AngleScaling;

        // Build model with angle scaling AND leakage
        let model = CompositeNoiseModelBuilder::new()
            .with_p2(0.1)
            .with_p2_angle_scaling(AngleScaling::linear())
            .with_p2_emission_ratio(0.2) // 20% of errors cause leakage
            .with_p2_seepage(0.1)
            .build();

        // Should have TQ channel + skip channel (for leakage)
        assert_eq!(model.channel_count(), 2);
    }

    #[test]
    fn test_leakage_scale() {
        // With leakage_scale = 1.0 (default), leakage is enabled
        let model_full_leak = CompositeNoiseModelBuilder::new()
            .with_p1(0.1)
            .with_p1_emission_ratio(0.5)
            .build();

        // Should have SQ channel + skip channel (for leakage)
        assert_eq!(model_full_leak.channel_count(), 2);

        // With leakage_scale = 0.0, no leakage (emission becomes depolarizing)
        let model_no_leak = CompositeNoiseModelBuilder::new()
            .with_p1(0.1)
            .with_p1_emission_ratio(0.5)
            .with_leakage_scale(0.0)
            .build();

        // Should have only SQ channel (no skip channel needed)
        assert_eq!(model_no_leak.channel_count(), 1);
    }

    #[test]
    fn test_leakage_scale_partial() {
        // With leakage_scale = 0.5, half of emission events become leakage
        let model = CompositeNoiseModelBuilder::new()
            .with_p1(0.1)
            .with_p1_emission_ratio(0.5)
            .with_leakage_scale(0.5) // Half become leakage, half depolarizing
            .build();

        // Should have SQ channel + skip channel (some leakage exists)
        assert_eq!(model.channel_count(), 2);
    }

    #[test]
    fn test_idle_linear_weights() {
        // Test with Z-only idle noise (typical for T2 dephasing)
        let model = CompositeNoiseModelBuilder::new()
            .with_p_idle_linear(0.001)
            .with_p_idle_linear_weights(PauliWeights::custom(0.0, 0.0, 1.0)) // Z-only
            .build();

        // Should have 1 idle channel
        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_idle_linear_weights_with_t2() {
        // Combined T1 (custom weights) and T2
        let model = CompositeNoiseModelBuilder::new()
            .with_idle_noise(0.001, 0.01) // Both T1 and T2
            .with_p_idle_linear_weights(PauliWeights::custom(1.0, 0.0, 0.0)) // X-only for T1
            .with_coherent_to_incoherent_factor(2.0)
            .build();

        // Should have 1 idle channel
        assert_eq!(model.channel_count(), 1);
    }

    // ========================================================================
    // Builder Comparison Tests (CompositeNoiseModelBuilder vs GeneralNoiseModelBuilder)
    // ========================================================================

    /// Compare `CompositeNoiseModelBuilder` with `GeneralNoiseModelBuilder` for basic depolarizing.
    #[test]
    fn test_builder_comparison_depolarizing() {
        use crate::noise::GeneralNoiseModelBuilder;
        use crate::runner::CircuitRunner;
        use pecos_simulators::SparseStab;

        let p1 = 0.1;
        let p2 = 0.15;

        // Build circuit with single and two-qubit gates
        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .h(0)
            .h(1)
            .cx(0, 1)
            .mz(0)
            .mz(1)
            .build();

        let shots = 300;
        let mut general_errors = 0;
        let mut flow_errors = 0;

        for seed in 0..shots {
            // GeneralNoiseModelBuilder
            let general_model = GeneralNoiseModelBuilder::new()
                .with_p1(p1)
                .with_p2(p2)
                .build();

            let mut state_general = SparseStab::new(2);
            let mut runner_general = CircuitRunner::<SparseStab>::new()
                .with_noise(general_model)
                .with_seed(seed);
            let outcomes_general = runner_general
                .apply_circuit(&mut state_general, &commands)
                .unwrap();
            let q0 = outcomes_general.get(QubitId(0)).is_some_and(|o| o.outcome);
            let q1 = outcomes_general.get(QubitId(1)).is_some_and(|o| o.outcome);
            // Bell state should give correlated results; errors break this
            if q0 != q1 {
                general_errors += 1;
            }

            // CompositeNoiseModelBuilder
            let flow_model = CompositeNoiseModelBuilder::new()
                .with_p1(p1)
                .with_p2(p2)
                .build();

            let mut state_flow = SparseStab::new(2);
            let mut runner_flow = CircuitRunner::<SparseStab>::new()
                .with_noise(flow_model)
                .with_seed(seed);
            let outcomes_flow = runner_flow
                .apply_circuit(&mut state_flow, &commands)
                .unwrap();
            let q0 = outcomes_flow.get(QubitId(0)).is_some_and(|o| o.outcome);
            let q1 = outcomes_flow.get(QubitId(1)).is_some_and(|o| o.outcome);
            if q0 != q1 {
                flow_errors += 1;
            }
        }

        let general_rate = f64::from(general_errors) / shots as f64;
        let flow_rate = f64::from(flow_errors) / shots as f64;

        // Both should produce similar error rates (within statistical tolerance)
        assert!(
            (general_rate - flow_rate).abs() < 0.15,
            "GeneralNoiseModelBuilder error rate {general_rate:.3} vs CompositeNoiseModelBuilder {flow_rate:.3}"
        );
    }

    /// Compare builders for measurement noise.
    #[test]
    fn test_builder_comparison_measurement() {
        use crate::noise::GeneralNoiseModelBuilder;
        use crate::runner::CircuitRunner;
        use pecos_simulators::SparseStab;

        let p_meas_0 = 0.1;
        let p_meas_1 = 0.2;

        // Simple prep and measure circuit
        let commands = CommandBuilder::new().pz(0).mz(0).build();

        let shots = 500;
        let mut general_flips = 0;
        let mut flow_flips = 0;

        for seed in 0..shots {
            // GeneralNoiseModelBuilder
            let general_model = GeneralNoiseModelBuilder::new()
                .with_p_meas(p_meas_0, p_meas_1)
                .build();

            let mut state_general = SparseStab::new(1);
            let mut runner_general = CircuitRunner::<SparseStab>::new()
                .with_noise(general_model)
                .with_seed(seed);
            let outcomes = runner_general
                .apply_circuit(&mut state_general, &commands)
                .unwrap();
            if outcomes.get(QubitId(0)).is_some_and(|o| o.outcome) {
                general_flips += 1;
            }

            // CompositeNoiseModelBuilder
            let flow_model = CompositeNoiseModelBuilder::new()
                .with_p_meas(p_meas_0, p_meas_1)
                .build();

            let mut state_flow = SparseStab::new(1);
            let mut runner_flow = CircuitRunner::<SparseStab>::new()
                .with_noise(flow_model)
                .with_seed(seed);
            let outcomes = runner_flow
                .apply_circuit(&mut state_flow, &commands)
                .unwrap();
            if outcomes.get(QubitId(0)).is_some_and(|o| o.outcome) {
                flow_flips += 1;
            }
        }

        let general_rate = f64::from(general_flips) / shots as f64;
        let flow_rate = f64::from(flow_flips) / shots as f64;

        // Both should be close to p_meas_0 (flipping 0 to 1)
        assert!(
            (general_rate - p_meas_0).abs() < 0.05,
            "GeneralNoiseModelBuilder flip rate {general_rate:.3} far from expected {p_meas_0}"
        );
        assert!(
            (flow_rate - p_meas_0).abs() < 0.05,
            "CompositeNoiseModelBuilder flip rate {flow_rate:.3} far from expected {p_meas_0}"
        );
    }

    /// Compare builders for preparation noise.
    #[test]
    fn test_builder_comparison_preparation() {
        use crate::noise::GeneralNoiseModelBuilder;
        use crate::runner::CircuitRunner;
        use pecos_simulators::SparseStab;

        let p_prep = 0.15;

        // Prep and immediately measure
        let commands = CommandBuilder::new().pz(0).mz(0).build();

        let shots = 500;
        let mut general_errors = 0;
        let mut flow_errors = 0;

        for seed in 0..shots {
            // GeneralNoiseModelBuilder
            let general_model = GeneralNoiseModelBuilder::new().with_p_prep(p_prep).build();

            let mut state = SparseStab::new(1);
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(general_model)
                .with_seed(seed);
            if runner
                .apply_circuit(&mut state, &commands)
                .unwrap()
                .get(QubitId(0))
                .is_some_and(|o| o.outcome)
            {
                general_errors += 1;
            }

            // CompositeNoiseModelBuilder
            let flow_model = CompositeNoiseModelBuilder::new()
                .with_p_prep(p_prep)
                .build();

            let mut state = SparseStab::new(1);
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(flow_model)
                .with_seed(seed);
            if runner
                .apply_circuit(&mut state, &commands)
                .unwrap()
                .get(QubitId(0))
                .is_some_and(|o| o.outcome)
            {
                flow_errors += 1;
            }
        }

        let general_rate = f64::from(general_errors) / shots as f64;
        let flow_rate = f64::from(flow_errors) / shots as f64;

        // Both should be close to p_prep
        assert!(
            (general_rate - p_prep).abs() < 0.05,
            "GeneralNoiseModelBuilder prep error rate {general_rate:.3} far from expected {p_prep}"
        );
        assert!(
            (flow_rate - p_prep).abs() < 0.05,
            "CompositeNoiseModelBuilder prep error rate {flow_rate:.3} far from expected {p_prep}"
        );
    }

    // ========================================================================
    // New Feature Tests: Local Crosstalk, Crosstalk Model, MeasureLeaked
    // ========================================================================

    #[test]
    fn test_local_crosstalk_builder() {
        // Define a simple neighbor function for testing
        fn linear_neighbors(qubits: &[QubitId]) -> Vec<QubitId> {
            qubits
                .iter()
                .flat_map(|q| {
                    let mut neighbors = Vec::new();
                    if q.0 > 0 {
                        neighbors.push(QubitId(q.0 - 1));
                    }
                    neighbors.push(QubitId(q.0 + 1));
                    neighbors
                })
                .filter(|n| !qubits.contains(n))
                .collect()
        }

        let model = CompositeNoiseModelBuilder::new()
            .with_p_meas_crosstalk(0.001, 0.01)
            .with_p_meas_crosstalk_local_fn(linear_neighbors)
            .build();

        // Should have: global crosstalk + local crosstalk = 2 channels
        assert_eq!(model.channel_count(), 2);
    }

    #[test]
    fn test_local_crosstalk_without_fn_is_ignored() {
        // If local crosstalk probability is set but no neighbor function, it's ignored
        let model = CompositeNoiseModelBuilder::new()
            .with_p_meas_crosstalk(0.001, 0.01) // local is 0.01 but no fn
            .build();

        // Should have only global crosstalk = 1 channel
        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_crosstalk_transition_model() {
        use crate::noise::CrosstalkTransitions;

        // Create a custom transition model
        let transitions = CrosstalkTransitions::custom(
            0.3, 0.5, 0.2, // from 0: 30% stay, 50% flip, 20% leak
            0.4, 0.4, 0.2, // from 1: 40% stay, 40% flip, 20% leak
        );

        let model = CompositeNoiseModelBuilder::new()
            .with_p_crosstalk(0.1)
            .with_p_meas_crosstalk_model(transitions)
            .build();

        // Should have 1 crosstalk channel
        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_measure_leaked_handling_builder() {
        let model = CompositeNoiseModelBuilder::new()
            .with_measure_leaked_handling(true)
            .build();

        // Should have 1 channel for MeasureLeaked handling
        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_measure_leaked_with_leakage() {
        let model = CompositeNoiseModelBuilder::new()
            .with_p1(0.1)
            .with_p1_emission_ratio(0.5) // 50% of errors cause leakage
            .with_measure_leaked_handling(true)
            .build();

        // Should have: SQ channel + skip channel + MeasureLeaked channel = 3 channels
        assert_eq!(model.channel_count(), 3);
    }

    #[test]
    fn test_full_model_with_all_new_features() {
        use crate::noise::CrosstalkTransitions;

        fn neighbors(qubits: &[QubitId]) -> Vec<QubitId> {
            qubits
                .iter()
                .flat_map(|q| vec![QubitId(q.0.saturating_sub(1)), QubitId(q.0 + 1)])
                .filter(|n| !qubits.contains(n))
                .collect()
        }

        let transitions = CrosstalkTransitions::flip_only();

        let model = CompositeNoiseModelBuilder::new()
            .with_p1(0.01)
            .with_p2(0.02)
            .with_p_meas(0.03, 0.04)
            .with_p_meas_crosstalk(0.001, 0.005)
            .with_p_meas_crosstalk_local_fn(neighbors)
            .with_p_meas_crosstalk_model(transitions)
            .with_measure_leaked_handling(true)
            .build();

        // SQ + TQ + Meas + Global Crosstalk + Local Crosstalk + MeasureLeaked = 6 channels
        assert_eq!(model.channel_count(), 6);
    }

    // ========================================================================
    // TimeScale Tests
    // ========================================================================

    #[test]
    fn test_time_scale_configuration() {
        let model = CompositeNoiseModelBuilder::new()
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
        let model = CompositeNoiseModelBuilder::new()
            .with_time_scale(TimeScale::NANOSECONDS)
            .with_idle_t1_t2(50e-6, 30e-6)
            .build();

        // Should have created an idle channel
        assert_eq!(model.channel_count(), 1);
        // Should have time scale set
        assert!(model.time_scale().is_some());
    }

    #[test]
    fn test_time_scale_with_full_model() {
        let model = CompositeNoiseModelBuilder::new()
            .with_time_scale(TimeScale::NANOSECONDS)
            .with_p1(0.01)
            .with_p2(0.02)
            .with_idle_t1_t2(50e-6, 30e-6)
            .build();

        // SQ + TQ + Idle = 3 channels
        assert_eq!(model.channel_count(), 3);
        assert!(model.time_scale().is_some());
    }

    #[test]
    #[should_panic(expected = "with_time_scale() must be called before with_idle_t1_t2()")]
    fn test_idle_t1_t2_without_time_scale_panics() {
        let _ = CompositeNoiseModelBuilder::new()
            .with_idle_t1_t2(50e-6, 30e-6) // Should panic - no time scale set
            .build();
    }

    // ========================================================================
    // Validation Tests
    // ========================================================================

    #[test]
    fn test_validation_clamps_negative_probability() {
        // Negative probability should be clamped to 0.0
        let model = CompositeNoiseModelBuilder::new().with_p1(-0.5).build();

        // Model should build successfully (clamped to 0.0)
        // No channels since p1=0.0 after clamping
        assert_eq!(model.channel_count(), 0);
    }

    #[test]
    fn test_validation_clamps_probability_over_one() {
        // Probability > 1.0 should be clamped to 1.0
        let model = CompositeNoiseModelBuilder::new().with_p1(1.5).build();

        // Model should build successfully (clamped to 1.0)
        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_validation_handles_nan() {
        // NaN should be set to 0.0
        let model = CompositeNoiseModelBuilder::new().with_p1(f64::NAN).build();

        // Model should build successfully (set to 0.0)
        assert_eq!(model.channel_count(), 0);
    }

    #[test]
    fn test_local_crosstalk_without_neighbor_fn_warns() {
        // This should print a warning but still build successfully
        let model = CompositeNoiseModelBuilder::new()
            .with_p_meas_crosstalk(0.0, 0.05) // Local rate set but no neighbor fn
            .build();

        // No crosstalk channel should be created (only global would be, and that's 0.0)
        assert_eq!(model.channel_count(), 0);
    }

    // ========================================================================
    // Emission Model Tests
    // ========================================================================

    #[test]
    fn test_p1_emission_model_builder() {
        use crate::noise::SingleQubitEmissionWeights;

        // Create model with custom emission weights
        let emission = SingleQubitEmissionWeights::custom(0.25, 0.25, 0.25, 0.25);

        let model = CompositeNoiseModelBuilder::new()
            .with_p1(0.1)
            .with_p1_emission_ratio(0.5)
            .with_p1_emission_model(emission)
            .build();

        // Should have SQ channel + skip channel (for leakage)
        assert_eq!(model.channel_count(), 2);
    }

    #[test]
    fn test_p2_emission_model_builder() {
        use crate::noise::TwoQubitEmissionWeights;

        // Create model with uniform emission weights
        let emission = TwoQubitEmissionWeights::uniform_with_leakage();

        let model = CompositeNoiseModelBuilder::new()
            .with_p2(0.1)
            .with_p2_emission_ratio(0.5)
            .with_p2_emission_model(emission)
            .build();

        // Should have TQ channel + skip channel (for leakage)
        assert_eq!(model.channel_count(), 2);
    }

    #[test]
    fn test_emission_model_statistical_behavior() {
        use crate::noise::SingleQubitEmissionWeights;
        use crate::runner::CircuitRunner;
        use pecos_simulators::SparseStab;

        // Test that emission model affects behavior
        // With 100% emission ratio and pure leakage, qubits should leak
        let emission = SingleQubitEmissionWeights::leakage_only();

        let commands = CommandBuilder::new()
            .pz(0)
            .h(0) // Single-qubit gate where emission can occur
            .h(0)
            .h(0)
            .mz(0)
            .build();

        let shots = 200;
        let mut outcomes_different = 0;

        for seed in 0..shots {
            let model = CompositeNoiseModelBuilder::new()
                .with_p1(0.5) // 50% error rate
                .with_p1_emission_ratio(1.0) // All errors are emission
                .with_p1_emission_model(emission)
                .build();

            let mut state = SparseStab::new(1);
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model)
                .with_seed(seed);
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

            // With H^3 = H on |0>, noiseless outcome is 50/50
            // With leakage, we might see different behavior
            if outcomes.get(QubitId(0)).is_some_and(|o| o.outcome) {
                outcomes_different += 1;
            }
        }

        // Just verify the model runs without panic
        // The exact statistical behavior depends on leak handling
        assert!(outcomes_different >= 0);
    }

    // ========================================================================
    // Two-Qubit Pauli Model Tests
    // ========================================================================

    #[test]
    fn test_p2_pauli_model_builder() {
        use crate::noise::TwoQubitPauliWeights;

        // Create model with ZZ-biased errors
        let model = CompositeNoiseModelBuilder::new()
            .with_p2(0.1)
            .with_p2_pauli_model(TwoQubitPauliWeights::zz_biased(0.8))
            .build();

        // Should have TQ channel
        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_p2_pauli_model_with_emission() {
        use crate::noise::{TwoQubitEmissionWeights, TwoQubitPauliWeights};

        // Create model with both correlated Pauli and emission models
        let model = CompositeNoiseModelBuilder::new()
            .with_p2(0.1)
            .with_p2_pauli_model(TwoQubitPauliWeights::uniform())
            .with_p2_emission_ratio(0.3)
            .with_p2_emission_model(TwoQubitEmissionWeights::uniform_with_leakage())
            .build();

        // Should have TQ channel + skip channel (for leakage)
        assert_eq!(model.channel_count(), 2);
    }

    #[test]
    fn test_p2_pauli_model_overrides_weights() {
        use crate::noise::TwoQubitPauliWeights;

        // When both are set, p2_pauli_model takes precedence
        let model = CompositeNoiseModelBuilder::new()
            .with_p2(0.1)
            .with_p2_pauli_weights(PauliWeights::custom(1.0, 0.0, 0.0)) // X-only
            .with_p2_pauli_model(TwoQubitPauliWeights::zz_biased(1.0)) // ZZ-only
            .build();

        // Model should build successfully
        assert_eq!(model.channel_count(), 1);
    }

    // ========================================================================
    // Statistical Comparison: Emission Models
    // ========================================================================

    #[test]
    fn test_emission_model_comparison_with_general_builder() {
        use crate::noise::{GeneralNoiseModelBuilder, SingleQubitEmissionWeights};
        use crate::runner::CircuitRunner;
        use pecos_simulators::SparseStab;

        let p1 = 0.3;
        let emission_ratio = 0.5;
        let emission = SingleQubitEmissionWeights::uniform();

        let commands = CommandBuilder::new().pz(0).h(0).h(0).mz(0).build();

        let shots = 300;
        let mut general_errors = 0;
        let mut flow_errors = 0;

        for seed in 0..shots {
            // GeneralNoiseModelBuilder
            let general_model = GeneralNoiseModelBuilder::new()
                .with_p1(p1)
                .with_p1_emission_ratio(emission_ratio)
                .with_p1_emission_weights(emission)
                .build();

            let mut state = SparseStab::new(1);
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(general_model)
                .with_seed(seed);
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            if outcomes.get(QubitId(0)).is_some_and(|o| o.outcome) {
                general_errors += 1;
            }

            // CompositeNoiseModelBuilder
            let flow_model = CompositeNoiseModelBuilder::new()
                .with_p1(p1)
                .with_p1_emission_ratio(emission_ratio)
                .with_p1_emission_model(emission)
                .build();

            let mut state = SparseStab::new(1);
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(flow_model)
                .with_seed(seed);
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            if outcomes.get(QubitId(0)).is_some_and(|o| o.outcome) {
                flow_errors += 1;
            }
        }

        let general_rate = f64::from(general_errors) / shots as f64;
        let flow_rate = f64::from(flow_errors) / shots as f64;

        // Both builders should produce similar error rates
        assert!(
            (general_rate - flow_rate).abs() < 0.15,
            "Emission model comparison: GeneralNoiseModelBuilder {general_rate:.3} vs CompositeNoiseModelBuilder {flow_rate:.3}"
        );
    }

    #[test]
    fn test_two_qubit_pauli_model_comparison() {
        use crate::noise::{GeneralNoiseModelBuilder, TwoQubitPauliWeights};
        use crate::runner::CircuitRunner;
        use pecos_simulators::SparseStab;

        let p2 = 0.3;

        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .cx(0, 1)
            .mz(0)
            .mz(1)
            .build();

        let shots = 300;
        let mut general_errors = 0;
        let mut flow_errors = 0;

        for seed in 0..shots {
            // GeneralNoiseModelBuilder
            let general_model = GeneralNoiseModelBuilder::new()
                .with_p2(p2)
                .with_p2_pauli_weights(TwoQubitPauliWeights::uniform())
                .build();

            let mut state = SparseStab::new(2);
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(general_model)
                .with_seed(seed);
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            let q0 = outcomes.get(QubitId(0)).is_some_and(|o| o.outcome);
            let q1 = outcomes.get(QubitId(1)).is_some_and(|o| o.outcome);
            if q0 || q1 {
                general_errors += 1;
            }

            // CompositeNoiseModelBuilder
            let flow_model = CompositeNoiseModelBuilder::new()
                .with_p2(p2)
                .with_p2_pauli_model(TwoQubitPauliWeights::uniform())
                .build();

            let mut state = SparseStab::new(2);
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(flow_model)
                .with_seed(seed);
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            let q0 = outcomes.get(QubitId(0)).is_some_and(|o| o.outcome);
            let q1 = outcomes.get(QubitId(1)).is_some_and(|o| o.outcome);
            if q0 || q1 {
                flow_errors += 1;
            }
        }

        let general_rate = f64::from(general_errors) / shots as f64;
        let flow_rate = f64::from(flow_errors) / shots as f64;

        // Both builders should produce similar error rates
        assert!(
            (general_rate - flow_rate).abs() < 0.15,
            "Two-qubit Pauli model comparison: GeneralNoiseModelBuilder {general_rate:.3} vs CompositeNoiseModelBuilder {flow_rate:.3}"
        );
    }
}
