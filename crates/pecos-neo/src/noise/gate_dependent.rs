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

//! Gate-dependent noise channel.
//!
//! Applies different error rates for different gate types, allowing
//! calibration-based noise models where different gates have different
//! fidelities.

use super::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse, PauliWeights};
use crate::command::{GateCommand, GateType};
use pecos_rng::PecosRng;
use rand::RngExt;
use smallvec::SmallVec;
use std::collections::HashMap;

/// Configuration for a single gate type's noise.
#[derive(Debug, Clone)]
pub struct GateNoiseConfig {
    /// Error probability for this gate type.
    pub error_probability: f64,
    /// Pauli weight distribution when an error occurs.
    pub pauli_weights: PauliWeights,
}

impl GateNoiseConfig {
    /// Create a new gate noise configuration with uniform Pauli distribution.
    #[must_use]
    pub fn new(error_probability: f64) -> Self {
        Self {
            error_probability,
            pauli_weights: PauliWeights::uniform(),
        }
    }

    /// Set custom Pauli weights.
    #[must_use]
    pub fn with_pauli_weights(mut self, weights: PauliWeights) -> Self {
        self.pauli_weights = weights;
        self
    }
}

/// Noise channel that applies different error rates based on gate type.
///
/// This allows modeling realistic quantum hardware where different gates
/// have different error rates (e.g., T gates typically have higher errors
/// than Clifford gates on some platforms).
///
/// # Example
///
/// ```
/// use pecos_neo::noise::{GateDependentChannel, GateNoiseConfig, PauliWeights};
/// use pecos_neo::command::GateType;
///
/// let channel = GateDependentChannel::new()
///     .with_gate(GateType::H, GateNoiseConfig::new(0.001))
///     .with_gate(GateType::T, GateNoiseConfig::new(0.01))
///     .with_gate(GateType::CX, GateNoiseConfig::new(0.02))
///     .with_default(0.005);  // Fallback for unlisted gates
/// ```
#[derive(Debug, Clone, Default)]
pub struct GateDependentChannel {
    /// Gate-specific configurations.
    gate_configs: HashMap<GateType, GateNoiseConfig>,
    /// Default configuration for gates not in the map.
    default_config: Option<GateNoiseConfig>,
}

impl GateDependentChannel {
    /// Create an empty gate-dependent channel.
    ///
    /// By default, no noise is applied. Use `with_gate` and `with_default`
    /// to configure error rates.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add noise configuration for a specific gate type.
    #[must_use]
    pub fn with_gate(mut self, gate_type: GateType, config: GateNoiseConfig) -> Self {
        self.gate_configs.insert(gate_type, config);
        self
    }

    /// Add a simple error probability for a specific gate type.
    ///
    /// Uses uniform Pauli distribution.
    #[must_use]
    pub fn with_gate_error(mut self, gate_type: GateType, error_probability: f64) -> Self {
        self.gate_configs
            .insert(gate_type, GateNoiseConfig::new(error_probability));
        self
    }

    /// Set the default configuration for gates not explicitly configured.
    #[must_use]
    pub fn with_default(mut self, error_probability: f64) -> Self {
        self.default_config = Some(GateNoiseConfig::new(error_probability));
        self
    }

    /// Set the default configuration with custom Pauli weights.
    #[must_use]
    pub fn with_default_config(mut self, config: GateNoiseConfig) -> Self {
        self.default_config = Some(config);
        self
    }

    /// Get the configuration for a specific gate type.
    fn get_config(&self, gate_type: GateType) -> Option<&GateNoiseConfig> {
        self.gate_configs
            .get(&gate_type)
            .or(self.default_config.as_ref())
    }
}

impl NoiseChannel for GateDependentChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        match event {
            NoiseEvent::AfterGate { gate_type, .. } => {
                // Only respond if we have a configuration for this gate type
                self.get_config(*gate_type)
                    .is_some_and(|c| c.error_probability > 0.0)
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

        let Some(config) = self.get_config(*gate_type) else {
            return NoiseResponse::None;
        };

        let mut gates = SmallVec::new();

        // Fast path: check if any leakage exists at all
        let has_any_leakage = ctx.leaked_count() > 0;

        for &qubit in *qubits {
            // Skip leaked qubits (fast path skips check if no leakage exists)
            if has_any_leakage && ctx.is_leaked(qubit) {
                continue;
            }

            // Apply error with probability
            if rng.random::<f64>() < config.error_probability {
                let pauli = config.pauli_weights.sample(rng.random::<f64>());
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
        "GateDependentChannel"
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
    fn test_gate_dependent_different_rates() {
        let channel = GateDependentChannel::new()
            .with_gate_error(GateType::H, 1.0)
            .with_gate_error(GateType::SZ, 0.0);

        let qubits = [QubitId(0)];
        let angles = [];

        // H gate should have noise
        let h_event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };
        assert!(channel.responds_to(&h_event));

        // SZ gate should have no noise (p=0)
        let sz_event = NoiseEvent::AfterGate {
            gate_type: GateType::SZ,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };
        assert!(!channel.responds_to(&sz_event));

        // X gate not configured - should not respond without default
        let x_event = NoiseEvent::AfterGate {
            gate_type: GateType::X,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };
        assert!(!channel.responds_to(&x_event));
    }

    #[test]
    fn test_gate_dependent_with_default() {
        let channel = GateDependentChannel::new()
            .with_gate_error(GateType::H, 0.5)
            .with_default(0.1);

        let qubits = [QubitId(0)];
        let angles = [];

        // H gate uses specific rate
        let h_event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };
        assert!(channel.responds_to(&h_event));

        // Unconfigured gate uses default
        let x_event = NoiseEvent::AfterGate {
            gate_type: GateType::X,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };
        assert!(channel.responds_to(&x_event));
    }

    #[test]
    fn test_gate_dependent_applies_noise() {
        let channel = GateDependentChannel::new().with_gate_error(GateType::H, 1.0);

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);

        // With p=1.0, should always inject a gate
        assert!(matches!(response, NoiseResponse::InjectGates(_)));
    }

    #[test]
    fn test_gate_dependent_custom_pauli_weights() {
        let channel = GateDependentChannel::new().with_gate(
            GateType::H,
            GateNoiseConfig::new(1.0).with_pauli_weights(PauliWeights::custom(0.0, 0.0, 1.0)),
        );

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Should always produce Z gate with Z-only weights
        if let NoiseResponse::InjectGates(gates) = response {
            assert_eq!(gates.len(), 1);
            assert_eq!(gates[0].gate_type, GateType::Z);
        } else {
            panic!("Expected InjectGates response");
        }
    }

    #[test]
    fn test_gate_dependent_skips_leaked_qubits() {
        let channel = GateDependentChannel::new().with_gate_error(GateType::H, 1.0);

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(0));
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Leaked qubits should not get noise
        assert!(response.is_none());
    }
}
