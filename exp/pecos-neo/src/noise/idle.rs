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
//! ## Coherent vs Incoherent Dephasing
//!
//! - **Coherent**: Deterministic RZ rotation with angle = rate * duration.
//!   Represents systematic phase errors.
//!
//! - **Incoherent**: Stochastic Z error with probability = sin(rate * duration)^2.
//!   Represents random dephasing.

use super::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse, PauliWeights};
use crate::command::{GateCommand, GateType};
use pecos_core::{Angle64, TimeUnits};
use pecos_rng::PecosRng;
use rand::RngExt;
use smallvec::SmallVec;

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

    /// Error rate per time unit for quadratic (dephasing) noise.
    ///
    /// For coherent: angle = `quadratic_rate` * duration.
    /// For incoherent: probability = sin(`quadratic_rate` * duration)^2.
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
}

impl Default for IdleChannel {
    fn default() -> Self {
        Self {
            linear_rate: 0.0,
            linear_weights: PauliWeights::custom(0.0, 0.0, 1.0), // Z-only by default
            quadratic_rate: 0.0,
            coherent_dephasing: false,
            coherent_to_incoherent_factor: 1.0,
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

    /// Calculate linear (stochastic) error probability for a given duration.
    fn linear_probability(&self, duration: TimeUnits) -> f64 {
        let t = duration.as_f64();
        (self.linear_rate * t).min(1.0)
    }

    /// Calculate quadratic dephasing probability (for incoherent mode).
    ///
    /// Applies the coherent-to-incoherent factor to compensate for
    /// not modeling coherent phase accumulation.
    fn quadratic_probability(&self, duration: TimeUnits) -> f64 {
        let t = duration.as_f64();
        let effective_rate = self.quadratic_rate * self.coherent_to_incoherent_factor;
        let angle = effective_rate * t;
        angle.sin().powi(2)
    }

    /// Calculate quadratic dephasing angle (for coherent mode).
    fn quadratic_angle(&self, duration: TimeUnits) -> f64 {
        let t = duration.as_f64();
        self.quadratic_rate * t
    }
}

impl NoiseChannel for IdleChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        if self.linear_rate <= 0.0 && self.quadratic_rate <= 0.0 {
            return false;
        }
        matches!(event, NoiseEvent::IdleTime { .. })
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let NoiseEvent::IdleTime { qubits, duration } = event else {
            return NoiseResponse::None;
        };

        let mut gates = SmallVec::new();

        // Fast path: check if any leakage exists at all
        let has_any_leakage = ctx.leaked_count() > 0;

        // Apply linear (stochastic) noise
        if self.linear_rate > 0.0 {
            let p_linear = self.linear_probability(*duration);
            for &qubit in *qubits {
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
                let angle = self.quadratic_angle(*duration);
                if angle.abs() > f64::EPSILON {
                    for &qubit in *qubits {
                        // Skip leaked qubits (fast path skips check if no leakage exists)
                        if !has_any_leakage || !ctx.is_leaked(qubit) {
                            gates.push(GateCommand::rz(qubit, Angle64::from_radians(angle)));
                        }
                    }
                }
            } else {
                // Incoherent dephasing: stochastic Z with sin^2 probability
                let p_quad = self.quadratic_probability(*duration);
                if p_quad > 0.0 {
                    for &qubit in *qubits {
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

        if gates.is_empty() {
            NoiseResponse::None
        } else {
            NoiseResponse::inject_gates(gates)
        }
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
        let p = channel.linear_probability(TimeUnits::new(10));
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
        // pi/2 rad/ns -> sin^2(pi/2) = 1
        let channel = IdleChannel {
            quadratic_rate: std::f64::consts::FRAC_PI_2,
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
        // With factor = 2.0 and rate = pi/4, effective rate = pi/2
        // sin^2(pi/2) = 1.0 -> always error
        let channel = IdleChannel {
            quadratic_rate: std::f64::consts::FRAC_PI_4,
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
}
