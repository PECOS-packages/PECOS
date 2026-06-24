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

//! Measurement noise channel.
//!
//! This is a traditional standalone channel implementation. For composable,
//! declarative noise models with conditional logic, see `CompositeChannel` in
//! `pecos_neo::noise::composite::prelude`.
//!
//! ## When to use this vs `CompositeChannel`
//!
//! **Use `MeasurementChannel` when:**
//! - You want simple symmetric or asymmetric readout error
//! - Performance is critical
//!
//! **Use `CompositeChannel` when:**
//! - You need outcome-dependent noise (different error on 0 vs 1)
//! - You need to compose with other noise effects
//! - You need leaked qubit handling integrated with measurement
//!
//! Handles asymmetric measurement errors (readout errors).

use super::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse};
use pecos_random::PecosRng;
use smallvec::SmallVec;

/// Noise channel for measurement operations.
///
/// Supports asymmetric measurement errors where the probability of
/// misreading 0 as 1 differs from misreading 1 as 0.
#[derive(Debug, Clone)]
pub struct MeasurementChannel {
    /// Probability of flipping a 0 measurement to 1.
    pub p_meas_0_to_1: f64,

    /// Probability of flipping a 1 measurement to 0.
    pub p_meas_1_to_0: f64,

    // Precomputed probability thresholds for fast sampling
    threshold_0_to_1: u64,
    threshold_1_to_0: u64,
}

impl Default for MeasurementChannel {
    fn default() -> Self {
        Self {
            p_meas_0_to_1: 0.0,
            p_meas_1_to_0: 0.0,
            threshold_0_to_1: 0,
            threshold_1_to_0: 0,
        }
    }
}

impl MeasurementChannel {
    /// Create a symmetric measurement error channel.
    #[must_use]
    pub fn symmetric(p: f64) -> Self {
        let threshold = PecosRng::probability_threshold(p);
        Self {
            p_meas_0_to_1: p,
            p_meas_1_to_0: p,
            threshold_0_to_1: threshold,
            threshold_1_to_0: threshold,
        }
    }

    /// Create an asymmetric measurement error channel.
    #[must_use]
    pub fn asymmetric(p_0_to_1: f64, p_1_to_0: f64) -> Self {
        Self {
            p_meas_0_to_1: p_0_to_1,
            p_meas_1_to_0: p_1_to_0,
            threshold_0_to_1: PecosRng::probability_threshold(p_0_to_1),
            threshold_1_to_0: PecosRng::probability_threshold(p_1_to_0),
        }
    }

    /// Check if this channel has any effect.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.p_meas_0_to_1 > 0.0 || self.p_meas_1_to_0 > 0.0
    }

    /// Scale both error probabilities by a factor.
    ///
    /// This multiplies both `p_meas_0_to_1` and `p_meas_1_to_0` by `scale`.
    /// Useful for globally adjusting noise levels.
    #[must_use]
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.p_meas_0_to_1 *= scale;
        self.p_meas_1_to_0 *= scale;
        self.threshold_0_to_1 = PecosRng::probability_threshold(self.p_meas_0_to_1);
        self.threshold_1_to_0 = PecosRng::probability_threshold(self.p_meas_1_to_0);
        self
    }
}

impl NoiseChannel for MeasurementChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        if !self.is_active() {
            return false;
        }
        matches!(event, NoiseEvent::AfterMeasurement { .. })
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let NoiseEvent::AfterMeasurement { qubits, outcomes } = event else {
            return NoiseResponse::None;
        };

        let mut flips = SmallVec::new();

        // Fast path: check if any leakage exists at all
        let has_any_leakage = ctx.leaked_count() > 0;

        for (&qubit, &outcome) in qubits.iter().zip(outcomes.iter()) {
            // Skip leaked qubits (fast path skips check if no leakage exists)
            if has_any_leakage && ctx.is_leaked(qubit) {
                continue;
            }

            // Apply asymmetric measurement error (using precomputed threshold)
            let threshold = if outcome {
                self.threshold_1_to_0
            } else {
                self.threshold_0_to_1
            };

            if rng.check_probability(threshold) {
                flips.push(qubit);
            }
        }

        if flips.is_empty() {
            NoiseResponse::None
        } else {
            NoiseResponse::FlipOutcomes(flips)
        }
    }

    /// Optimized combined check + apply.
    #[inline]
    fn try_apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> Option<NoiseResponse> {
        // Early exit if no errors configured
        if !self.is_active() {
            return None;
        }

        let NoiseEvent::AfterMeasurement { qubits, outcomes } = event else {
            return None;
        };

        let mut flips = SmallVec::new();

        // Fast path: check if any leakage exists at all
        let has_any_leakage = ctx.leaked_count() > 0;

        for (&qubit, &outcome) in qubits.iter().zip(outcomes.iter()) {
            // Skip leaked qubits (fast path skips check if no leakage exists)
            if has_any_leakage && ctx.is_leaked(qubit) {
                continue;
            }

            // Apply asymmetric measurement error (using precomputed threshold)
            let threshold = if outcome {
                self.threshold_1_to_0
            } else {
                self.threshold_0_to_1
            };

            if rng.check_probability(threshold) {
                flips.push(qubit);
            }
        }

        if flips.is_empty() {
            Some(NoiseResponse::None)
        } else {
            Some(NoiseResponse::FlipOutcomes(flips))
        }
    }

    fn name(&self) -> &'static str {
        "MeasurementChannel"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

/// Measurement error realized as a physical X flip of the qubit state
/// just before readout.
///
/// Unlike [`MeasurementChannel`], which flips only the classical record
/// (the post-measurement state is untouched), this channel injects an X
/// gate before the measurement executes, so the error persists in the
/// state and propagates to later operations on the same qubit. This is
/// the convention of the engines stack's depolarizing noise family and
/// of the DEM builder's measurement mechanisms: measuring the same qubit
/// twice without a reset sees the second outcome flipped at rate
/// `2p(1-p)`, not `p`.
///
/// State flips are inherently symmetric (X swaps |0> and |1>), so there
/// is no asymmetric variant; use [`MeasurementChannel`] for asymmetric
/// readout-record errors.
#[derive(Debug, Clone)]
pub struct MeasurementStateFlipChannel {
    /// Probability of an X flip before each measurement.
    pub p: f64,
    /// Precomputed probability threshold for fast sampling.
    threshold: u64,
}

impl MeasurementStateFlipChannel {
    /// Create a state-flip measurement error channel.
    #[must_use]
    pub fn new(p: f64) -> Self {
        Self {
            p,
            threshold: PecosRng::probability_threshold(p),
        }
    }
}

impl NoiseChannel for MeasurementStateFlipChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        if self.p <= 0.0 {
            return false;
        }
        matches!(event, NoiseEvent::BeforeMeasurement { .. })
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let NoiseEvent::BeforeMeasurement { qubits } = event else {
            return NoiseResponse::None;
        };

        let mut gates: SmallVec<[crate::command::GateCommand; 4]> = SmallVec::new();
        let has_any_leakage = ctx.leaked_count() > 0;

        for &qubit in *qubits {
            if has_any_leakage && ctx.is_leaked(qubit) {
                continue;
            }
            if rng.check_probability(self.threshold) {
                gates.push(crate::command::GateCommand::new(
                    crate::command::GateType::X,
                    smallvec::smallvec![qubit],
                ));
            }
        }

        if gates.is_empty() {
            NoiseResponse::None
        } else {
            NoiseResponse::inject_gates(gates)
        }
    }

    fn name(&self) -> &'static str {
        "MeasurementStateFlipChannel"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::QubitId;

    /// The state-flip vs record-flip distinction: measuring the same
    /// qubit twice without a reset. A record flip leaves the state
    /// untouched (second outcome flips at p); a state flip persists
    /// (second outcome flips at 2p(1-p)).
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn state_flip_propagates_to_second_measurement_record_flip_does_not() {
        use crate::prelude::*;
        use pecos_simulators::SparseStab;

        const SHOTS: usize = 20_000;
        let p = 0.25;

        let commands = CommandBuilder::new().pz(&[0]).mz(&[0]).mz(&[0]).build();

        let second_meas_rate = |model: ComposableNoiseModel| -> f64 {
            let mut state = SparseStab::new(1);
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(model)
                .with_seed(42);
            let mut ones = 0usize;
            for _ in 0..SHOTS {
                state.reset();
                let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
                let bits: Vec<bool> = outcomes.iter().map(|o| o.outcome).collect();
                assert_eq!(bits.len(), 2);
                if bits[1] {
                    ones += 1;
                }
            }
            ones as f64 / SHOTS as f64
        };

        let five_sigma = |q: f64| 5.0 * (q * (1.0 - q) / SHOTS as f64).sqrt();

        let state_flip = second_meas_rate(
            ComposableNoiseModel::new().add_channel(MeasurementStateFlipChannel::new(p)),
        );
        let expected_state = 2.0 * p * (1.0 - p);
        assert!(
            (state_flip - expected_state).abs() < five_sigma(expected_state),
            "state flip second-measure rate: got {state_flip}, expected {expected_state}"
        );

        let record_flip = second_meas_rate(
            ComposableNoiseModel::new().add_channel(MeasurementChannel::symmetric(p)),
        );
        assert!(
            (record_flip - p).abs() < five_sigma(p),
            "record flip second-measure rate: got {record_flip}, expected {p}"
        );
    }

    #[test]
    fn test_symmetric_measurement_error() {
        let channel = MeasurementChannel::symmetric(1.0); // Always flip

        let qubits = [QubitId(0)];
        let outcomes = [false]; // Measured 0
        let event = NoiseEvent::AfterMeasurement {
            qubits: &qubits,
            outcomes: &outcomes,
        };

        assert!(channel.responds_to(&event));

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);
        assert!(matches!(response, NoiseResponse::FlipOutcomes(_)));
    }

    #[test]
    fn test_asymmetric_measurement_error() {
        // Only flip 0->1, never 1->0
        let channel = MeasurementChannel::asymmetric(1.0, 0.0);

        let qubits = [QubitId(0), QubitId(1)];
        let outcomes = [false, true]; // 0 and 1

        let event = NoiseEvent::AfterMeasurement {
            qubits: &qubits,
            outcomes: &outcomes,
        };

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Only qubit 0 should flip (measured 0, and p_0_to_1 = 1.0)
        if let NoiseResponse::FlipOutcomes(flips) = response {
            assert_eq!(flips.len(), 1);
            assert_eq!(flips[0], QubitId(0));
        } else {
            panic!("Expected FlipOutcomes response");
        }
    }

    #[test]
    fn test_no_flip_on_zero_probability() {
        let channel = MeasurementChannel::symmetric(0.0);

        let qubits = [QubitId(0)];
        let outcomes = [false];
        let event = NoiseEvent::AfterMeasurement {
            qubits: &qubits,
            outcomes: &outcomes,
        };

        assert!(!channel.responds_to(&event));
    }
}
