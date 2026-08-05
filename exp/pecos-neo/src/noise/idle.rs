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

//! Idle noise channel.
//!
//! This is a traditional standalone channel implementation. For composable,
//! declarative noise models with conditional logic, see `CompositeChannel` in
//! `pecos_neo::noise::composite::prelude`.
//!
//! ## When to use this vs `CompositeChannel`
//!
//! **Use `IdleChannel` when:**
//! - You want standard T1/T2 decay with linear/quadratic scaling
//! - Performance is critical (batched processing)
//!
//! **Use `CompositeChannel` when:**
//! - You need conditional idle noise (different for leaked qubits)
//! - You want to combine T1, T2, and ZZ crosstalk
//! - You need custom time-dependent behavior
//!
//! Handles T1/T2 decay and dephasing during idle time.
//!
//! ## Time Units
//!
//! All rates are specified per abstract time unit. The interpretation of time units
//! (nanoseconds, clock cycles, etc.) is defined by the noise model configuration.
//!
//! ## Noise Components
//!
//! - **Linear noise**: Stochastic errors with probability proportional to time.
//!   Models T1-like relaxation.
//!
//! - **Quadratic noise**: Can be coherent (RZ rotations) or incoherent (stochastic Z).
//!   Models T2-like dephasing.
//!
//! - **Sine-squared noise**: Independent stochastic X, Y, Z, or leakage events with
//!   per-axis probability `sin(rate * multiplier * duration)^2`.
//!
//! ## Coherent vs Incoherent Dephasing
//!
//! - **Coherent**: Deterministic RZ rotation with angle = rate * duration.
//!   Represents systematic phase errors.
//!
//! - **Incoherent**: Stochastic Z error with probability = sin(rate * duration / 2)^2.
//!   This is the exact Pauli twirl of the coherent RZ rotation.

use super::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse, PauliWeights};
use crate::command::{GateCommand, GateType};
use pecos_core::{Angle64, TimeUnits};
use pecos_random::PecosRng;
use rand::RngExt;
use smallvec::SmallVec;
use std::collections::BTreeMap;

/// Noise channel for idle time (memory errors).
///
/// Models T1 relaxation and T2 dephasing during idle periods.
/// Rates are specified per abstract time unit.
#[derive(Debug, Clone)]
pub struct IdleChannel {
    /// Error rate per time unit for linear (stochastic) noise.
    ///
    /// Probability of error = `linear_rate` * duration.
    pub linear_rate: f64,

    /// Distribution of Pauli errors for linear noise.
    ///
    /// By default, uses Z-only errors. Can be set to uniform for depolarizing
    /// or any custom distribution.
    pub linear_weights: PauliWeights,

    /// DEM-style stochastic sine-squared idle rate in radians per time unit.
    pub sin_squared_rate: f64,

    /// Unnormalized per-axis relative multipliers for the sine-squared idle family.
    pub sin_squared_model: BTreeMap<String, f64>,

    /// Error rate per time unit for quadratic (dephasing) noise.
    ///
    /// For coherent: angle = `quadratic_rate` * duration.
    /// For incoherent: probability = sin(`quadratic_rate` * duration / 2)^2.
    ///
    /// The factor of one half makes the incoherent model the exact Pauli twirl
    /// of the coherent RZ rotation. This deliberately changes numerical results
    /// from earlier versions for incoherent quadratic idle noise.
    pub quadratic_rate: f64,

    /// Whether to model quadratic dephasing coherently (RZ) or incoherently (stochastic Z).
    pub coherent_dephasing: bool,

    /// Scaling factor to convert coherent dephasing rates to incoherent rates.
    ///
    /// When using incoherent (stochastic) dephasing, this factor adjusts the
    /// dephasing rate. This is a fudge factor used to artificially increase
    /// the dephasing rate when modeling quadratic dephasing stochastically,
    /// since such modeling does not account for coherent effects.
    ///
    /// Default is 1.0 (no adjustment). Values > 1.0 increase the effective
    /// incoherent dephasing rate.
    pub coherent_to_incoherent_factor: f64,

    /// Duration of the idle-noise site applied after a two-qubit gate.
    ///
    /// A duration of zero disables after-two-qubit idle sites. When enabled,
    /// the same linear, quadratic, and sine-squared mechanisms used for
    /// explicit idle events are applied to every distinct gate operand.
    pub idle_after_2q: f64,
}

impl Default for IdleChannel {
    fn default() -> Self {
        Self {
            linear_rate: 0.0,
            linear_weights: PauliWeights::custom(0.0, 0.0, 1.0), // Z-only by default
            sin_squared_rate: 0.0,
            sin_squared_model: BTreeMap::new(),
            quadratic_rate: 0.0,
            coherent_dephasing: false,
            coherent_to_incoherent_factor: 1.0,
            idle_after_2q: 0.0,
        }
    }
}

impl IdleChannel {
    /// Create an idle noise channel with linear time dependence.
    ///
    /// Rate is per abstract time unit.
    #[must_use]
    pub fn linear(rate_per_time_unit: f64) -> Self {
        Self {
            linear_rate: rate_per_time_unit,
            ..Default::default()
        }
    }

    /// Create an idle noise channel with T1/T2 parameters in abstract time units.
    ///
    /// # Arguments
    /// * `t1` - T1 relaxation time in time units
    /// * `t2` - T2 dephasing time in time units
    #[must_use]
    pub fn from_t1_t2(t1: f64, t2: f64) -> Self {
        // Approximate error rate from T1/T2
        // This is a simplified model
        let linear_rate = 1.0 / t1.max(1.0);
        let quadratic_rate = 1.0 / (t2 * t2).max(1.0);

        Self {
            linear_rate,
            quadratic_rate,
            ..Default::default()
        }
    }

    /// Set whether to use coherent dephasing.
    #[must_use]
    pub fn with_coherent_dephasing(mut self, coherent: bool) -> Self {
        self.coherent_dephasing = coherent;
        self
    }

    /// Set the linear noise Pauli weights.
    ///
    /// By default, linear noise is Z-only. Use this to set a custom distribution.
    #[must_use]
    pub fn with_linear_weights(mut self, weights: PauliWeights) -> Self {
        self.linear_weights = weights;
        self
    }

    /// Set linear noise to uniform depolarizing (X, Y, Z with equal probability).
    #[must_use]
    pub fn with_linear_depolarizing(mut self) -> Self {
        self.linear_weights = PauliWeights::uniform();
        self
    }

    /// Set the coherent-to-incoherent conversion factor.
    ///
    /// This factor is applied to the quadratic dephasing rate when using
    /// incoherent (stochastic) dephasing. It compensates for the fact that
    /// stochastic modeling doesn't capture coherent phase accumulation.
    ///
    /// Default is 1.0. Values > 1.0 increase the effective dephasing rate.
    #[must_use]
    pub fn with_coherent_to_incoherent_factor(mut self, factor: f64) -> Self {
        self.coherent_to_incoherent_factor = factor;
        self
    }

    /// Set the duration of the idle-noise site after each two-qubit gate.
    ///
    /// The duration uses the channel's abstract time units. A duration of zero
    /// disables these sites.
    #[must_use]
    pub fn with_idle_after_2q(mut self, duration: f64) -> Self {
        self.idle_after_2q = duration;
        self
    }

    /// Calculate linear (stochastic) error probability for a given duration.
    fn linear_probability(&self, duration: f64) -> f64 {
        (self.linear_rate * duration).min(1.0)
    }

    /// Calculate quadratic dephasing probability (for incoherent mode).
    ///
    /// Applies the coherent-to-incoherent factor as a multiplier on the rate,
    /// then uses the exact Pauli-twirl probability `sin(effective_angle / 2)^2`.
    /// This deliberately changes numerical results from earlier versions,
    /// which omitted the factor of one half.
    fn quadratic_probability(&self, duration: f64) -> f64 {
        let effective_angle = self.quadratic_rate * self.coherent_to_incoherent_factor * duration;
        (effective_angle / 2.0).sin().powi(2)
    }

    /// Calculate quadratic dephasing angle (for coherent mode).
    fn quadratic_angle(&self, duration: f64) -> f64 {
        self.quadratic_rate * duration
    }

    /// Calculate one axis's DEM-style sine-squared error probability.
    fn sin_squared_probability(rate: f64, multiplier: f64, duration: f64) -> f64 {
        (rate * multiplier * duration).sin().powi(2)
    }

    /// Apply every configured idle mechanism for one duration.
    fn apply_for_duration(
        &self,
        qubits: &[pecos_core::QubitId],
        duration: f64,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        if duration <= 0.0
            || (self.linear_rate <= 0.0
                && self.quadratic_rate <= 0.0
                && self.sin_squared_rate <= 0.0)
        {
            return NoiseResponse::None;
        }

        // A batched two-qubit command can contain multiple pairs. Preserve the
        // operand order while applying one idle site to each distinct qubit.
        let mut unique_qubits = SmallVec::<[pecos_core::QubitId; 4]>::new();
        for &qubit in qubits {
            if !unique_qubits.contains(&qubit) {
                unique_qubits.push(qubit);
            }
        }

        let mut gates = SmallVec::new();
        let mut leaked = SmallVec::new();

        // Fast path: check if any leakage exists at all
        let has_any_leakage = ctx.leaked_count() > 0;

        // Apply linear (stochastic) noise
        if self.linear_rate > 0.0 {
            let p_linear = self.linear_probability(duration);
            for &qubit in &unique_qubits {
                // Skip leaked qubits (fast path skips check if no leakage exists)
                if (!has_any_leakage || !ctx.is_leaked(qubit)) && rng.random::<f64>() < p_linear {
                    // Sample Pauli error from linear weights
                    let pauli = self.linear_weights.sample(rng.random::<f64>());
                    gates.push(GateCommand::new(pauli, smallvec::smallvec![qubit]));
                }
            }
        }

        // Apply quadratic (dephasing) noise
        if self.quadratic_rate > 0.0 {
            if self.coherent_dephasing {
                // Coherent dephasing: deterministic RZ rotation
                let angle = self.quadratic_angle(duration);
                if angle.abs() > f64::EPSILON {
                    for &qubit in &unique_qubits {
                        // Skip leaked qubits (fast path skips check if no leakage exists)
                        if !has_any_leakage || !ctx.is_leaked(qubit) {
                            gates.push(GateCommand::rz(qubit, Angle64::from_radians(angle)));
                        }
                    }
                }
            } else {
                // Incoherent dephasing: stochastic Z with exact Pauli-twirl probability
                let p_quad = self.quadratic_probability(duration);
                if p_quad > 0.0 {
                    for &qubit in &unique_qubits {
                        // Skip leaked qubits (fast path skips check if no leakage exists)
                        if (!has_any_leakage || !ctx.is_leaked(qubit))
                            && rng.random::<f64>() < p_quad
                        {
                            gates.push(GateCommand::new(GateType::Z, smallvec::smallvec![qubit]));
                        }
                    }
                }
            }
        }

        // Apply the DEM-style stochastic sine-squared family independently per axis.
        if self.sin_squared_rate > 0.0 {
            for axis in ["X", "Y", "Z", "L"] {
                let Some(multiplier) = self.sin_squared_model.get(axis).copied() else {
                    continue;
                };
                let probability =
                    Self::sin_squared_probability(self.sin_squared_rate, multiplier, duration);
                if probability <= f64::EPSILON {
                    continue;
                }

                for &qubit in &unique_qubits {
                    if (!has_any_leakage || !ctx.is_leaked(qubit))
                        && rng.random::<f64>() < probability
                    {
                        match axis {
                            "X" => gates
                                .push(GateCommand::new(GateType::X, smallvec::smallvec![qubit])),
                            "Y" => gates
                                .push(GateCommand::new(GateType::Y, smallvec::smallvec![qubit])),
                            "Z" => gates
                                .push(GateCommand::new(GateType::Z, smallvec::smallvec![qubit])),
                            "L" => leaked.push(qubit),
                            _ => unreachable!("sine-family model was validated by the builder"),
                        }
                    }
                }
            }
        }

        let response = if gates.is_empty() {
            NoiseResponse::None
        } else {
            NoiseResponse::inject_gates(gates)
        };
        if leaked.is_empty() {
            response
        } else {
            response.combine(NoiseResponse::MarkLeaked(leaked))
        }
    }
}

impl NoiseChannel for IdleChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        if self.linear_rate <= 0.0 && self.quadratic_rate <= 0.0 && self.sin_squared_rate <= 0.0 {
            return false;
        }
        match event {
            NoiseEvent::IdleTime { duration, .. } => *duration != TimeUnits::ZERO,
            NoiseEvent::AfterGate { gate_type, .. } => {
                self.idle_after_2q > 0.0 && gate_type.is_two_qubit()
            }
            _ => false,
        }
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let (qubits, duration) = match event {
            NoiseEvent::IdleTime { qubits, duration } => (*qubits, duration.as_f64()),
            NoiseEvent::AfterGate {
                gate_type, qubits, ..
            } if self.idle_after_2q > 0.0 && gate_type.is_two_qubit() => {
                (*qubits, self.idle_after_2q)
            }
            _ => return NoiseResponse::None,
        };

        self.apply_for_duration(qubits, duration, ctx, rng)
    }

    fn name(&self) -> &'static str {
        "IdleChannel"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::QubitId;

    fn collect_gates(response: NoiseResponse) -> Vec<GateCommand> {
        match response {
            NoiseResponse::InjectGates(gates) => (*gates).into_vec(),
            NoiseResponse::Multiple(responses) => {
                responses.into_iter().flat_map(collect_gates).collect()
            }
            _ => Vec::new(),
        }
    }

    fn collect_leaked(response: NoiseResponse) -> Vec<QubitId> {
        match response {
            NoiseResponse::MarkLeaked(qubits) => qubits.into_vec(),
            NoiseResponse::Multiple(responses) => {
                responses.into_iter().flat_map(collect_leaked).collect()
            }
            _ => Vec::new(),
        }
    }

    fn after_cx(qubits: &[QubitId]) -> NoiseEvent<'_> {
        NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits,
            angles: &[],
            gate_id: None,
        }
    }

    #[test]
    fn test_idle_error() {
        let channel = IdleChannel::linear(1.0); // 100% error per ns

        let qubits = [QubitId(0)];
        let duration = TimeUnits::new(1);
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration,
        };

        assert!(channel.responds_to(&event));

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);

        if let NoiseResponse::InjectGates(gates) = response {
            assert_eq!(gates.len(), 1);
            assert_eq!(gates[0].gate_type, GateType::Z);
        } else {
            panic!("Expected InjectGates response");
        }
    }

    #[test]
    fn test_no_error_at_zero_rate() {
        let channel = IdleChannel::default();

        let qubits = [QubitId(0)];
        let duration = TimeUnits::new(1000);
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration,
        };

        assert!(!channel.responds_to(&event));
    }

    #[test]
    fn test_linear_probability_scaling() {
        let channel = IdleChannel::linear(0.001);

        // At 10ns: p = 0.001 * 10 = 0.01
        let p = channel.linear_probability(TimeUnits::new(10).as_f64());
        assert!((p - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_linear_with_custom_weights() {
        // X-biased linear noise
        let channel =
            IdleChannel::linear(1.0).with_linear_weights(PauliWeights::custom(1.0, 0.0, 0.0));

        let qubits = [QubitId(0)];
        let duration = TimeUnits::new(1);
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration,
        };

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);

        // With X-only weights, should produce X gate
        if let NoiseResponse::InjectGates(gates) = response {
            assert_eq!(gates.len(), 1);
            assert_eq!(gates[0].gate_type, GateType::X);
        } else {
            panic!("Expected InjectGates response");
        }
    }

    #[test]
    fn test_linear_depolarizing() {
        // Uniform linear noise (depolarizing)
        let channel = IdleChannel::linear(1.0).with_linear_depolarizing();

        // linear_weights should be uniform
        let weights = channel.linear_weights;
        assert!((weights.x - 1.0 / 3.0).abs() < 1e-10);
        assert!((weights.y - 1.0 / 3.0).abs() < 1e-10);
        assert!((weights.z - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherent_dephasing() {
        let channel = IdleChannel::default().with_coherent_dephasing(true);
        let channel = IdleChannel {
            quadratic_rate: 1.0, // 1 rad/ns
            ..channel
        };

        let qubits = [QubitId(0)];
        let duration = TimeUnits::new(1);
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration,
        };

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Should produce an RZ gate with angle 1.0 rad
        if let NoiseResponse::InjectGates(gates) = response {
            assert_eq!(gates.len(), 1);
            assert_eq!(gates[0].gate_type, GateType::RZ);
            assert!((gates[0].angles[0].to_radians() - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected InjectGates response");
        }
    }

    #[test]
    fn test_incoherent_dephasing() {
        // pi rad/ns -> sin^2(pi/2) = 1
        let channel = IdleChannel {
            quadratic_rate: std::f64::consts::PI,
            ..Default::default()
        };

        let qubits = [QubitId(0)];
        let duration = TimeUnits::new(1);
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration,
        };

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);

        // With sin^2(pi/2) = 1.0 probability, should always produce Z gate
        if let NoiseResponse::InjectGates(gates) = response {
            assert_eq!(gates.len(), 1);
            assert_eq!(gates[0].gate_type, GateType::Z);
        } else {
            panic!("Expected InjectGates response");
        }
    }

    #[test]
    fn test_coherent_to_incoherent_factor() {
        // With factor = 2.0 and rate = pi/2, effective angle = pi
        // sin^2(pi/2) = 1.0 -> always error
        let channel = IdleChannel {
            quadratic_rate: std::f64::consts::FRAC_PI_2,
            coherent_to_incoherent_factor: 2.0,
            ..Default::default()
        };

        let qubits = [QubitId(0)];
        let duration = TimeUnits::new(1);
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration,
        };

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);

        // With effective sin^2(pi/2) = 1.0, should always produce Z gate
        if let NoiseResponse::InjectGates(gates) = response {
            assert_eq!(gates.len(), 1);
            assert_eq!(gates[0].gate_type, GateType::Z);
        } else {
            panic!("Expected InjectGates response");
        }
    }

    #[test]
    fn sine_probability_matches_engines_numeric_value() {
        let probability = IdleChannel::sin_squared_probability(0.03, 1.0, 10.0);
        assert!((probability - 0.087_332_192_545_160_84).abs() < f64::EPSILON);
    }

    #[test]
    fn sine_application_uses_rate_multiplier_and_duration() {
        let channel = IdleChannel {
            sin_squared_rate: 0.03,
            sin_squared_model: BTreeMap::from([("X".to_string(), 2.0)]),
            ..Default::default()
        };
        let qubits = std::array::from_fn::<_, 32, _>(QubitId);
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration: TimeUnits::new(5),
        };
        let expected_probability = 0.087_332_192_545_160_84;
        let mut expected_rng = PecosRng::seed_from_u64(3);
        let expected_qubits = qubits
            .iter()
            .copied()
            .filter(|_| expected_rng.random::<f64>() < expected_probability)
            .collect::<Vec<_>>();
        let expected_next = expected_rng.random::<u64>();

        let mut actual_rng = PecosRng::seed_from_u64(3);
        let actual_gates =
            collect_gates(channel.apply(&event, &mut NoiseContext::new(), &mut actual_rng));
        assert_eq!(
            actual_gates
                .iter()
                .map(|gate| gate.qubits[0])
                .collect::<Vec<_>>(),
            expected_qubits
        );
        assert!(
            actual_gates
                .iter()
                .all(|gate| gate.gate_type == GateType::X)
        );
        assert_eq!(actual_rng.random::<u64>(), expected_next);
    }

    #[test]
    fn x_weighted_sine_model_emits_x_not_z() {
        let channel = IdleChannel {
            sin_squared_rate: std::f64::consts::FRAC_PI_2,
            sin_squared_model: BTreeMap::from([("X".to_string(), 1.0)]),
            ..Default::default()
        };
        let qubits = [QubitId(0)];
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration: TimeUnits::new(1),
        };
        let gates = collect_gates(channel.apply(
            &event,
            &mut NoiseContext::new(),
            &mut PecosRng::seed_from_u64(5),
        ));

        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::X);
    }

    #[test]
    fn sine_model_axes_are_independent_in_xyzl_order() {
        let channel = IdleChannel {
            sin_squared_rate: std::f64::consts::FRAC_PI_2,
            sin_squared_model: BTreeMap::from([
                ("X".to_string(), 1.0),
                ("Y".to_string(), 1.0),
                ("Z".to_string(), 1.0),
                ("L".to_string(), 1.0),
            ]),
            ..Default::default()
        };
        let qubits = [QubitId(0)];
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration: TimeUnits::new(1),
        };
        let response = channel.apply(
            &event,
            &mut NoiseContext::new(),
            &mut PecosRng::seed_from_u64(7),
        );
        let gates = collect_gates(response.clone());

        assert_eq!(
            gates.iter().map(|gate| gate.gate_type).collect::<Vec<_>>(),
            [GateType::X, GateType::Y, GateType::Z]
        );
        assert_eq!(collect_leaked(response), [QubitId(0)]);
    }

    #[test]
    fn after_2q_duration_scales_linear_noise() {
        let qubits = std::array::from_fn::<_, 64, _>(QubitId);
        let short = IdleChannel::linear(0.25).with_idle_after_2q(1.0);
        let long = IdleChannel::linear(0.25).with_idle_after_2q(4.0);

        let mut short_rng = PecosRng::seed_from_u64(17);
        let short_gates = collect_gates(short.apply(
            &after_cx(&qubits),
            &mut NoiseContext::new(),
            &mut short_rng,
        ));

        let mut long_rng = PecosRng::seed_from_u64(17);
        let long_gates =
            collect_gates(long.apply(&after_cx(&qubits), &mut NoiseContext::new(), &mut long_rng));

        assert!(short_gates.len() < qubits.len());
        assert_eq!(long_gates.len(), qubits.len());
    }

    #[test]
    fn quadratic_only_noise_reaches_after_2q_sites() {
        let channel = IdleChannel {
            quadratic_rate: std::f64::consts::PI,
            idle_after_2q: 1.0,
            ..Default::default()
        };
        let qubits = [QubitId(0), QubitId(1)];
        let mut rng = PecosRng::seed_from_u64(8);

        let gates =
            collect_gates(channel.apply(&after_cx(&qubits), &mut NoiseContext::new(), &mut rng));

        assert_eq!(gates.len(), 2);
        assert!(gates.iter().all(|gate| gate.gate_type == GateType::Z));
    }

    #[test]
    fn linear_weights_are_honored_at_after_2q_sites() {
        let channel = IdleChannel::linear(1.0)
            .with_linear_weights(PauliWeights::custom(1.0, 0.0, 0.0))
            .with_idle_after_2q(1.0);
        let qubits = [QubitId(0), QubitId(1)];
        let mut rng = PecosRng::seed_from_u64(23);

        let gates =
            collect_gates(channel.apply(&after_cx(&qubits), &mut NoiseContext::new(), &mut rng));

        assert_eq!(gates.len(), 2);
        assert!(gates.iter().all(|gate| gate.gate_type == GateType::X));
    }

    #[test]
    fn batched_after_2q_idles_every_distinct_operand() {
        let channel = IdleChannel::linear(1.0)
            .with_linear_weights(PauliWeights::custom(1.0, 0.0, 0.0))
            .with_idle_after_2q(1.0);
        let qubits = [QubitId(0), QubitId(1), QubitId(2), QubitId(3), QubitId(1)];
        let mut rng = PecosRng::seed_from_u64(29);

        let gates =
            collect_gates(channel.apply(&after_cx(&qubits), &mut NoiseContext::new(), &mut rng));
        let affected = gates.iter().map(|gate| gate.qubits[0]).collect::<Vec<_>>();

        assert_eq!(
            affected,
            vec![QubitId(0), QubitId(1), QubitId(2), QubitId(3)]
        );
    }

    #[test]
    fn after_2q_channel_ignores_single_qubit_gates() {
        let channel = IdleChannel::linear(1.0).with_idle_after_2q(1.0);
        let qubits = [QubitId(0)];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &[],
            gate_id: None,
        };

        assert!(!channel.responds_to(&event));
        let mut actual_rng = PecosRng::seed_from_u64(31);
        assert!(
            channel
                .apply(&event, &mut NoiseContext::new(), &mut actual_rng)
                .is_none()
        );
        let mut expected_rng = PecosRng::seed_from_u64(31);
        assert_eq!(actual_rng.random::<u64>(), expected_rng.random::<u64>());
    }

    #[test]
    fn zero_duration_and_zero_rates_produce_nothing_without_rng_draws() {
        let qubits = [QubitId(0), QubitId(1)];

        let zero_duration = IdleChannel::linear(1.0);
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration: TimeUnits::ZERO,
        };
        assert!(!zero_duration.responds_to(&event));
        let mut actual_rng = PecosRng::seed_from_u64(31);
        assert!(
            zero_duration
                .apply(&event, &mut NoiseContext::new(), &mut actual_rng)
                .is_none()
        );
        let mut expected_rng = PecosRng::seed_from_u64(31);
        assert_eq!(actual_rng.random::<u64>(), expected_rng.random::<u64>());

        let zero_after_2q_duration = IdleChannel::linear(1.0).with_idle_after_2q(0.0);
        let event = after_cx(&qubits);
        assert!(!zero_after_2q_duration.responds_to(&event));
        let mut actual_rng = PecosRng::seed_from_u64(37);
        assert!(
            zero_after_2q_duration
                .apply(&event, &mut NoiseContext::new(), &mut actual_rng)
                .is_none()
        );
        let mut expected_rng = PecosRng::seed_from_u64(37);
        assert_eq!(actual_rng.random::<u64>(), expected_rng.random::<u64>());

        let zero_rates = IdleChannel::default().with_idle_after_2q(10.0);
        let event = after_cx(&qubits);
        assert!(!zero_rates.responds_to(&event));
        let mut actual_rng = PecosRng::seed_from_u64(41);
        assert!(
            zero_rates
                .apply(&event, &mut NoiseContext::new(), &mut actual_rng)
                .is_none()
        );
        let mut expected_rng = PecosRng::seed_from_u64(41);
        assert_eq!(actual_rng.random::<u64>(), expected_rng.random::<u64>());

        let zero_sine_rate = IdleChannel {
            sin_squared_model: BTreeMap::from([("X".to_string(), 1.0)]),
            ..Default::default()
        };
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration: TimeUnits::new(1),
        };
        assert!(!zero_sine_rate.responds_to(&event));
        let mut actual_rng = PecosRng::seed_from_u64(43);
        assert!(
            zero_sine_rate
                .apply(&event, &mut NoiseContext::new(), &mut actual_rng)
                .is_none()
        );
        let mut expected_rng = PecosRng::seed_from_u64(43);
        assert_eq!(actual_rng.random::<u64>(), expected_rng.random::<u64>());

        let nonzero_sine_rate = IdleChannel {
            sin_squared_rate: std::f64::consts::FRAC_PI_2,
            sin_squared_model: BTreeMap::from([("X".to_string(), 1.0)]),
            ..Default::default()
        };
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration: TimeUnits::ZERO,
        };
        let mut actual_rng = PecosRng::seed_from_u64(47);
        assert!(
            nonzero_sine_rate
                .apply(&event, &mut NoiseContext::new(), &mut actual_rng)
                .is_none()
        );
        let mut expected_rng = PecosRng::seed_from_u64(47);
        assert_eq!(actual_rng.random::<u64>(), expected_rng.random::<u64>());
    }

    #[test]
    fn after_2q_noise_reproduces_exactly_for_the_same_seed() {
        let channel = IdleChannel::linear(0.4)
            .with_linear_depolarizing()
            .with_idle_after_2q(2.0);
        let qubits = std::array::from_fn::<_, 16, _>(QubitId);

        let sample = || {
            let mut rng = PecosRng::seed_from_u64(43);
            collect_gates(channel.apply(&after_cx(&qubits), &mut NoiseContext::new(), &mut rng))
        };

        assert_eq!(sample(), sample());
    }

    #[test]
    fn sine_noise_reproduces_exactly_for_the_same_seed() {
        let channel = IdleChannel {
            sin_squared_rate: 0.6,
            sin_squared_model: BTreeMap::from([
                ("X".to_string(), 0.5),
                ("Y".to_string(), 0.75),
                ("Z".to_string(), 1.0),
            ]),
            ..Default::default()
        };
        let qubits = std::array::from_fn::<_, 16, _>(QubitId);
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration: TimeUnits::new(1),
        };

        let sample = || {
            collect_gates(channel.apply(
                &event,
                &mut NoiseContext::new(),
                &mut PecosRng::seed_from_u64(53),
            ))
        };

        let first = sample();
        assert!(!first.is_empty());
        assert_eq!(first, sample());
    }

    #[test]
    fn incoherent_quadratic_probability_is_exact_twirl_of_coherent_angle() {
        let theta = 1.0;
        let incoherent = IdleChannel {
            quadratic_rate: theta,
            coherent_to_incoherent_factor: 1.0,
            ..Default::default()
        };
        let probability = incoherent.quadratic_probability(1.0);
        assert!((probability - 0.229_848_847_065_930_15).abs() < 1e-15);

        let coherent = IdleChannel {
            coherent_dephasing: true,
            ..incoherent
        };
        let qubits = [QubitId(0)];
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration: TimeUnits::new(1),
        };
        let mut rng = PecosRng::seed_from_u64(47);
        let gates = collect_gates(coherent.apply(&event, &mut NoiseContext::new(), &mut rng));

        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::RZ);
        assert!((gates[0].angles[0].to_radians() - theta).abs() < 1e-15);
    }

    #[test]
    fn legacy_quadratic_paths_keep_their_pre_change_output_exactly() {
        let qubits = std::array::from_fn::<_, 8, _>(QubitId);
        let event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration: TimeUnits::new(2),
        };
        let incoherent = IdleChannel {
            quadratic_rate: 0.7,
            coherent_to_incoherent_factor: 1.3,
            ..Default::default()
        };
        let mut incoherent_rng = PecosRng::seed_from_u64(424);
        let incoherent_outputs = (0..4)
            .map(|_| {
                collect_gates(incoherent.apply(
                    &event,
                    &mut NoiseContext::new(),
                    &mut incoherent_rng,
                ))
            })
            .collect::<Vec<_>>();
        let expected_incoherent_qubits: [&[usize]; 4] = [
            &[1, 3, 4, 5, 7],
            &[0, 1, 3, 4, 5, 7],
            &[0, 1, 4, 6, 7],
            &[0, 1, 2, 4, 6, 7],
        ];
        let expected_incoherent = expected_incoherent_qubits
            .iter()
            .map(|qubits| {
                qubits
                    .iter()
                    .map(|&qubit| {
                        GateCommand::new(GateType::Z, smallvec::smallvec![QubitId(qubit)])
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            incoherent_outputs, expected_incoherent,
            "the complete incoherent gate payload changed"
        );
        assert_eq!(incoherent_rng.random::<u64>(), 13_820_570_602_603_389_690);

        let coherent = IdleChannel {
            coherent_dephasing: true,
            ..incoherent
        };
        let mut coherent_rng = PecosRng::seed_from_u64(424);
        let coherent_output =
            collect_gates(coherent.apply(&event, &mut NoiseContext::new(), &mut coherent_rng));
        let expected_coherent = qubits
            .iter()
            .map(|&qubit| GateCommand::rz(qubit, Angle64::from_radians(1.4)))
            .collect::<Vec<_>>();
        assert_eq!(
            coherent_output, expected_coherent,
            "the complete coherent gate payload changed"
        );
        assert_eq!(coherent_rng.random::<u64>(), 15_629_358_259_572_395_946);
    }
}
