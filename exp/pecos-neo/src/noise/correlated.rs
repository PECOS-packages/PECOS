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

//! Correlated noise channel.
//!
//! Models spatially correlated errors where an error on one qubit
//! increases the probability of error on nearby qubits. This is
//! common in physical systems with spatial proximity or crosstalk.

use super::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse, PauliWeights};
use crate::command::GateCommand;
use pecos_random::PecosRng;
use rand::RngExt;
use smallvec::SmallVec;

/// Noise channel with spatial correlations between qubits.
///
/// When an error occurs on one qubit during a multi-qubit gate, nearby qubits
/// have a correlated probability of also experiencing an error.
///
/// The correlation model works as follows:
/// 1. For each qubit, independently determine if a base error occurs
/// 2. If any qubit had an error, increase the probability for remaining qubits
///
/// # Correlation Factor
///
/// The `correlation` parameter controls how errors spread:
/// - `0.0` = Independent errors (standard depolarizing)
/// - `0.5` = 50% chance errors spread to neighbors
/// - `1.0` = Errors always spread to all qubits in the gate
///
/// # Example
///
/// ```
/// use pecos_neo::noise::CorrelatedNoiseChannel;
///
/// // 1% base error rate, 50% correlation
/// let channel = CorrelatedNoiseChannel::new(0.01, 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct CorrelatedNoiseChannel {
    /// Base error probability (before correlation effects).
    pub base_error_probability: f64,
    /// Correlation factor: how much errors on one qubit affect others.
    /// Range: [0.0, 1.0] where 0.0 = independent, 1.0 = fully correlated.
    pub correlation: f64,
    /// Pauli weight distribution for errors.
    pub pauli_weights: PauliWeights,
    /// Whether to apply correlated errors to single-qubit gates.
    /// When true, multiple consecutive 1Q gates can have correlated errors.
    pub single_qubit_correlation: bool,
}

impl Default for CorrelatedNoiseChannel {
    fn default() -> Self {
        Self {
            base_error_probability: 0.0,
            correlation: 0.0,
            pauli_weights: PauliWeights::uniform(),
            single_qubit_correlation: false,
        }
    }
}

impl CorrelatedNoiseChannel {
    /// Create a new correlated noise channel.
    ///
    /// # Arguments
    /// * `base_error_probability` - Base error rate per qubit
    /// * `correlation` - How much errors spread (0.0 = independent, 1.0 = fully correlated)
    #[must_use]
    pub fn new(base_error_probability: f64, correlation: f64) -> Self {
        Self {
            base_error_probability,
            correlation: correlation.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Set custom Pauli weights for the error distribution.
    #[must_use]
    pub fn with_pauli_weights(mut self, weights: PauliWeights) -> Self {
        self.pauli_weights = weights;
        self
    }

    /// Enable correlation for single-qubit gates.
    #[must_use]
    pub fn with_single_qubit_correlation(mut self) -> Self {
        self.single_qubit_correlation = true;
        self
    }

    /// Create a Z-correlated dephasing channel.
    ///
    /// Errors are Z (dephasing) only, with spatial correlation.
    #[must_use]
    pub fn correlated_dephasing(base_error_probability: f64, correlation: f64) -> Self {
        Self::new(base_error_probability, correlation)
            .with_pauli_weights(PauliWeights::custom(0.0, 0.0, 1.0))
    }

    /// Calculate the conditional error probability given whether a correlated qubit had an error.
    fn conditional_probability(&self, other_had_error: bool) -> f64 {
        if other_had_error {
            // Increase probability due to correlation
            self.base_error_probability + (1.0 - self.base_error_probability) * self.correlation
        } else {
            // Decrease probability (inverse correlation effect)
            self.base_error_probability * (1.0 - self.correlation)
        }
    }
}

impl NoiseChannel for CorrelatedNoiseChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        if self.base_error_probability <= 0.0 {
            return false;
        }
        match event {
            NoiseEvent::AfterGate {
                gate_type, qubits, ..
            } => {
                // For single-qubit gates, only respond if single_qubit_correlation is enabled
                if gate_type.is_single_qubit() {
                    self.single_qubit_correlation
                } else {
                    // Always respond to multi-qubit gates
                    qubits.len() >= 2
                }
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
        let NoiseEvent::AfterGate {
            gate_type, qubits, ..
        } = event
        else {
            return NoiseResponse::None;
        };

        // Skip noiseless gates
        if ctx.is_noiseless(*gate_type) {
            return NoiseResponse::None;
        }

        let mut gates = SmallVec::new();
        let mut any_error = false;

        // Fast path: check if any leakage exists at all
        let has_any_leakage = ctx.leaked_count() > 0;

        // Process qubits in order, with correlation from previous qubits
        for (i, &qubit) in qubits.iter().enumerate() {
            // Skip leaked qubits (fast path skips check if no leakage exists)
            if has_any_leakage && ctx.is_leaked(qubit) {
                continue;
            }

            // Calculate probability based on correlation with previous qubits
            let prob = if i == 0 {
                self.base_error_probability
            } else {
                self.conditional_probability(any_error)
            };

            // Apply error with calculated probability
            if rng.random::<f64>() < prob {
                any_error = true;
                let pauli = self.pauli_weights.sample(rng.random::<f64>());
                gates.push(GateCommand::new(pauli, smallvec::smallvec![qubit]));
            }
        }

        if gates.is_empty() {
            NoiseResponse::None
        } else {
            NoiseResponse::inject_gates(gates)
        }
    }

    fn name(&self) -> &'static str {
        "CorrelatedNoiseChannel"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

/// Statistics for analyzing correlated noise behavior.
#[derive(Debug, Clone, Default)]
pub struct CorrelationStats {
    /// Total number of gate events processed.
    pub total_events: usize,
    /// Number of events with at least one error.
    pub events_with_error: usize,
    /// Number of events where multiple qubits had errors.
    pub events_with_multiple_errors: usize,
    /// Total errors on first qubit.
    pub first_qubit_errors: usize,
    /// Total errors on second qubit.
    pub second_qubit_errors: usize,
}

impl CorrelationStats {
    /// Calculate the correlation coefficient from observed statistics.
    ///
    /// Returns a value between -1 and 1:
    /// - Positive: errors tend to occur together
    /// - Zero: independent errors
    /// - Negative: errors tend to be anti-correlated
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // rate calculation
    pub fn observed_correlation(&self) -> f64 {
        if self.total_events == 0 {
            return 0.0;
        }

        let p1 = self.first_qubit_errors as f64 / self.total_events as f64;
        let p2 = self.second_qubit_errors as f64 / self.total_events as f64;
        let p12 = self.events_with_multiple_errors as f64 / self.total_events as f64;

        // Calculate correlation: Cov(X1, X2) / (Std(X1) * Std(X2))
        let cov = p12 - p1 * p2;
        let std1 = (p1 * (1.0 - p1)).sqrt();
        let std2 = (p2 * (1.0 - p2)).sqrt();

        if std1 > 0.0 && std2 > 0.0 {
            cov / (std1 * std2)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::GateType;
    use pecos_core::QubitId;

    #[test]
    fn test_correlated_channel_creation() {
        let channel = CorrelatedNoiseChannel::new(0.1, 0.5);
        assert!((channel.base_error_probability - 0.1).abs() < 1e-10);
        assert!((channel.correlation - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_correlated_channel_clamping() {
        // Correlation should be clamped to [0, 1]
        let channel = CorrelatedNoiseChannel::new(0.1, 1.5);
        assert!((channel.correlation - 1.0).abs() < 1e-10);

        let channel = CorrelatedNoiseChannel::new(0.1, -0.5);
        assert!(channel.correlation.abs() < 1e-10);
    }

    #[test]
    fn test_correlated_responds_to_two_qubit_gates() {
        let channel = CorrelatedNoiseChannel::new(0.1, 0.5);

        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];

        let event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        assert!(channel.responds_to(&event));
    }

    #[test]
    fn test_correlated_ignores_single_qubit_by_default() {
        let channel = CorrelatedNoiseChannel::new(0.1, 0.5);

        let qubits = [QubitId(0)];
        let angles = [];

        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        assert!(!channel.responds_to(&event));
    }

    #[test]
    fn test_correlated_single_qubit_when_enabled() {
        let channel = CorrelatedNoiseChannel::new(0.1, 0.5).with_single_qubit_correlation();

        let qubits = [QubitId(0)];
        let angles = [];

        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        assert!(channel.responds_to(&event));
    }

    #[test]
    fn test_conditional_probability() {
        let channel = CorrelatedNoiseChannel::new(0.1, 0.5);

        // When no previous error
        let prob_no_error = channel.conditional_probability(false);
        assert!((prob_no_error - 0.05).abs() < 1e-10); // 0.1 * (1 - 0.5) = 0.05

        // When previous error occurred
        let prob_with_error = channel.conditional_probability(true);
        assert!((prob_with_error - 0.55).abs() < 1e-10); // 0.1 + (1 - 0.1) * 0.5 = 0.55
    }

    #[test]
    fn test_zero_correlation_is_independent() {
        let channel = CorrelatedNoiseChannel::new(0.1, 0.0);

        // With zero correlation, probability should be the same regardless of other errors
        let prob_no_error = channel.conditional_probability(false);
        let prob_with_error = channel.conditional_probability(true);

        assert!((prob_no_error - 0.1).abs() < 1e-10);
        assert!((prob_with_error - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_full_correlation() {
        let channel = CorrelatedNoiseChannel::new(0.1, 1.0);

        // With full correlation:
        // - If first had error, second is certain to have error
        // - If first had no error, second has zero probability
        let prob_no_error = channel.conditional_probability(false);
        let prob_with_error = channel.conditional_probability(true);

        assert!(prob_no_error.abs() < 1e-10);
        assert!((prob_with_error - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_stats() {
        let stats = CorrelationStats {
            total_events: 1000,
            first_qubit_errors: 100,
            second_qubit_errors: 100,
            events_with_multiple_errors: 50,
            events_with_error: 150,
        };

        // With these stats, there's positive correlation
        let corr = stats.observed_correlation();
        assert!(corr > 0.0, "Expected positive correlation, got {corr}");
    }

    #[test]
    fn test_correlated_skips_leaked_qubits() {
        let channel = CorrelatedNoiseChannel::new(1.0, 0.5);

        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(0));
        ctx.mark_leaked(QubitId(1));
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Both qubits leaked, should be no noise
        assert!(response.is_none());
    }
}
