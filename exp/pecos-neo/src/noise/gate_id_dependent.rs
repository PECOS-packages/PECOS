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

//! Gate ID-dependent noise channel.
//!
//! Applies different error rates based on `GateId`, allowing uniform noise
//! configuration for both core gates and custom gates. This is the preferred
//! approach for extensible gate systems where custom gates should have their
//! own noise rates.
//!
//! # Example
//!
//! ```
//! use pecos_neo::noise::{GateIdDependentChannel, GateIdNoiseConfig, PauliWeights};
//! use pecos_neo::extensible::{gates, GateId};
//!
//! let channel = GateIdDependentChannel::new()
//!     // Configure core gates by their GateId
//!     .with_gate(gates::H, GateIdNoiseConfig::new(0.001))
//!     .with_gate(gates::CX, GateIdNoiseConfig::new(0.02))
//!     // Configure custom gates the same way
//!     .with_gate(GateId(256), GateIdNoiseConfig::new(0.005))
//!     // Fallback for unlisted gates
//!     .with_default(0.001);
//! ```

use super::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse, PauliWeights};
use crate::command::{GateCommand, GateType};
use crate::extensible::GateId;
use pecos_random::PecosRng;
use rand::RngExt;
use smallvec::SmallVec;
use std::collections::HashMap;

/// Configuration for a single gate's noise, identified by `GateId`.
#[derive(Debug, Clone)]
pub struct GateIdNoiseConfig {
    /// Error probability for this gate.
    pub error_probability: f64,
    /// Pauli weight distribution when an error occurs.
    pub pauli_weights: PauliWeights,
}

impl GateIdNoiseConfig {
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

/// Noise channel that applies different error rates based on `GateId`.
///
/// Unlike `GateDependentChannel` which uses `GateType`, this channel uses
/// `GateId` for uniform handling of both core and custom gates. Every gate
/// (both core and custom) has a unique `GateId`, making this the preferred
/// approach for extensible gate systems.
///
/// # Example
///
/// ```
/// use pecos_neo::noise::{GateIdDependentChannel, GateIdNoiseConfig};
/// use pecos_neo::extensible::{gates, GateId};
///
/// // Core gates and custom gates can be configured the same way
/// let channel = GateIdDependentChannel::new()
///     .with_gate_error(gates::H, 0.001)      // H gate at 0.1% error
///     .with_gate_error(gates::CX, 0.02)     // CX gate at 2% error
///     .with_gate_error(GateId(256), 0.005) // Custom gate at 0.5% error
///     .with_default(0.001);                  // Everything else at 0.1%
/// ```
#[derive(Debug, Clone, Default)]
pub struct GateIdDependentChannel {
    /// Gate-specific configurations keyed by `GateId`.
    gate_configs: HashMap<GateId, GateIdNoiseConfig>,
    /// Default configuration for gates not in the map.
    default_config: Option<GateIdNoiseConfig>,
}

impl GateIdDependentChannel {
    /// Create an empty gate ID-dependent channel.
    ///
    /// By default, no noise is applied. Use `with_gate` and `with_default`
    /// to configure error rates.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add noise configuration for a specific gate ID.
    #[must_use]
    pub fn with_gate(mut self, gate_id: GateId, config: GateIdNoiseConfig) -> Self {
        self.gate_configs.insert(gate_id, config);
        self
    }

    /// Add a simple error probability for a specific gate ID.
    ///
    /// Uses uniform Pauli distribution.
    #[must_use]
    pub fn with_gate_error(mut self, gate_id: GateId, error_probability: f64) -> Self {
        self.gate_configs
            .insert(gate_id, GateIdNoiseConfig::new(error_probability));
        self
    }

    /// Configure a core gate by its `GateType`.
    ///
    /// This is a convenience method that converts the `GateType` to `GateId`.
    #[must_use]
    pub fn with_gate_type(self, gate_type: GateType, config: GateIdNoiseConfig) -> Self {
        self.with_gate(gate_type.to_gate_id(), config)
    }

    /// Configure a core gate by its `GateType` with a simple error probability.
    ///
    /// This is a convenience method that converts the `GateType` to `GateId`.
    #[must_use]
    pub fn with_gate_type_error(self, gate_type: GateType, error_probability: f64) -> Self {
        self.with_gate_error(gate_type.to_gate_id(), error_probability)
    }

    /// Set the default configuration for gates not explicitly configured.
    #[must_use]
    pub fn with_default(mut self, error_probability: f64) -> Self {
        self.default_config = Some(GateIdNoiseConfig::new(error_probability));
        self
    }

    /// Set the default configuration with custom Pauli weights.
    #[must_use]
    pub fn with_default_config(mut self, config: GateIdNoiseConfig) -> Self {
        self.default_config = Some(config);
        self
    }

    /// Get the configuration for a specific gate ID.
    fn get_config(&self, gate_id: GateId) -> Option<&GateIdNoiseConfig> {
        self.gate_configs
            .get(&gate_id)
            .or(self.default_config.as_ref())
    }
}

impl NoiseChannel for GateIdDependentChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        match event {
            NoiseEvent::AfterGate { gate_id, .. } => {
                // Only respond if we have a configuration for this gate ID
                gate_id
                    .and_then(|id| self.get_config(id))
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
            gate_type,
            qubits,
            gate_id,
            ..
        } = event
        else {
            return NoiseResponse::None;
        };

        // Skip noiseless gates
        if ctx.is_noiseless(*gate_type) {
            return NoiseResponse::None;
        }

        // Get gate_id - should always be Some for events from standard constructors
        let Some(id) = gate_id else {
            return NoiseResponse::None;
        };

        let Some(config) = self.get_config(*id) else {
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
        "GateIdDependentChannel"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extensible::gates;
    use pecos_core::QubitId;

    #[test]
    fn test_gate_id_dependent_different_rates() {
        let channel = GateIdDependentChannel::new()
            .with_gate_error(gates::H, 1.0)
            .with_gate_error(gates::SZ, 0.0);

        let qubits = [QubitId(0)];
        let angles = [];

        // H gate should have noise
        let h_event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: Some(gates::H),
        };
        assert!(channel.responds_to(&h_event));

        // SZ gate should have no noise (p=0)
        let sz_event = NoiseEvent::AfterGate {
            gate_type: GateType::SZ,
            qubits: &qubits,
            angles: &angles,
            gate_id: Some(gates::SZ),
        };
        assert!(!channel.responds_to(&sz_event));

        // X gate not configured - should not respond without default
        let x_event = NoiseEvent::AfterGate {
            gate_type: GateType::X,
            qubits: &qubits,
            angles: &angles,
            gate_id: Some(gates::X),
        };
        assert!(!channel.responds_to(&x_event));
    }

    #[test]
    fn test_gate_id_dependent_with_default() {
        let channel = GateIdDependentChannel::new()
            .with_gate_error(gates::H, 0.5)
            .with_default(0.1);

        let qubits = [QubitId(0)];
        let angles = [];

        // H gate uses specific rate
        let h_event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: Some(gates::H),
        };
        assert!(channel.responds_to(&h_event));

        // Unconfigured gate uses default
        let x_event = NoiseEvent::AfterGate {
            gate_type: GateType::X,
            qubits: &qubits,
            angles: &angles,
            gate_id: Some(gates::X),
        };
        assert!(channel.responds_to(&x_event));
    }

    #[test]
    fn test_gate_id_dependent_custom_gate() {
        // Custom gate with ID >= 256
        let custom_gate = GateId(300);

        let channel = GateIdDependentChannel::new()
            .with_gate_error(custom_gate, 1.0)
            .with_gate_error(gates::H, 0.0);

        let qubits = [QubitId(0)];
        let angles = [];

        // Custom gate should have noise
        // Note: custom gates don't have a GateType, so we use I as placeholder
        let custom_event = NoiseEvent::AfterGate {
            gate_type: GateType::I,
            qubits: &qubits,
            angles: &angles,
            gate_id: Some(custom_gate),
        };
        assert!(channel.responds_to(&custom_event));

        // H gate should not (p=0)
        let h_event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: Some(gates::H),
        };
        assert!(!channel.responds_to(&h_event));
    }

    #[test]
    fn test_gate_id_dependent_applies_noise() {
        let channel = GateIdDependentChannel::new().with_gate_error(gates::H, 1.0);

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: Some(gates::H),
        };

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);

        // With p=1.0, should always inject a gate
        assert!(matches!(response, NoiseResponse::InjectGates(_)));
    }

    #[test]
    fn test_gate_id_dependent_convenience_methods() {
        // Test with_gate_type convenience method
        let channel = GateIdDependentChannel::new()
            .with_gate_type_error(GateType::H, 0.5)
            .with_gate_type_error(GateType::CX, 0.02);

        let qubits = [QubitId(0)];
        let angles = [];

        let h_event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: Some(gates::H),
        };
        assert!(channel.responds_to(&h_event));
    }

    #[test]
    fn test_gate_id_dependent_skips_leaked_qubits() {
        let channel = GateIdDependentChannel::new().with_gate_error(gates::H, 1.0);

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: Some(gates::H),
        };

        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(0));
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Leaked qubits should not get noise
        assert!(response.is_none());
    }
}
