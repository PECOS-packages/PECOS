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

//! Category-based noise channel.
//!
//! Applies noise based on gate category (single-qubit, two-qubit, etc.).
//! Uses `GateDefinitions` from `NoiseContext` for category lookup.
//!
//! # Example
//!
//! ```
//! use pecos_neo::noise::{CategoryBasedChannel, ComposableNoiseModel};
//! use pecos_neo::extensible::{GateCategory, GateDefinitions};
//!
//! let gates = GateDefinitions::new();
//!
//! let noise = ComposableNoiseModel::new()
//!     .with_gate_definitions(gates)
//!     .add_channel(CategoryBasedChannel::new()
//!         .with_category(GateCategory::SingleQubitUnitary, 0.001)
//!         .with_category(GateCategory::TwoQubitUnitary, 0.01));
//! ```

use super::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse, PauliWeights};
use crate::command::GateCommand;
use crate::extensible::GateCategory;
use pecos_random::PecosRng;
use rand::RngExt;
use smallvec::SmallVec;

/// Configuration for a category's noise.
#[derive(Debug, Clone)]
struct CategoryConfig {
    /// Error probability for this category.
    error_probability: f64,
    /// Pauli weight distribution when an error occurs.
    pauli_weights: PauliWeights,
}

/// Noise channel that applies different error rates based on gate category.
///
/// Uses `GateDefinitions` from `NoiseContext` to determine gate categories.
/// This enables uniform noise configuration for both core and custom gates
/// based on their semantic category.
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::CategoryBasedChannel;
/// # use pecos_neo::extensible::GateCategory;
/// let channel = CategoryBasedChannel::new()
///     .with_category(GateCategory::SingleQubitUnitary, 0.001)
///     .with_category(GateCategory::TwoQubitUnitary, 0.01)
///     .with_category(GateCategory::Measurement, 0.005);
/// ```
#[derive(Debug, Clone, Default)]
pub struct CategoryBasedChannel {
    /// Per-category error rates. Index matches `category_to_index`.
    category_configs: [Option<CategoryConfig>; 8],
    /// Default error rate for unconfigured categories.
    default_error: Option<f64>,
}

impl CategoryBasedChannel {
    /// Create a new empty category-based channel.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set error probability for a category.
    #[must_use]
    pub fn with_category(mut self, category: GateCategory, error_probability: f64) -> Self {
        self.category_configs[category_to_index(category)] = Some(CategoryConfig {
            error_probability,
            pauli_weights: PauliWeights::uniform(),
        });
        self
    }

    /// Set error probability and Pauli weights for a category.
    #[must_use]
    pub fn with_category_weights(
        mut self,
        category: GateCategory,
        error_probability: f64,
        pauli_weights: PauliWeights,
    ) -> Self {
        self.category_configs[category_to_index(category)] = Some(CategoryConfig {
            error_probability,
            pauli_weights,
        });
        self
    }

    /// Set default error probability for unconfigured categories.
    #[must_use]
    pub fn with_default(mut self, error_probability: f64) -> Self {
        self.default_error = Some(error_probability);
        self
    }

    /// Get config for a category.
    fn get_config(&self, category: GateCategory) -> Option<&CategoryConfig> {
        self.category_configs[category_to_index(category)].as_ref()
    }

    /// Get error probability for a category.
    fn error_for_category(&self, category: GateCategory) -> f64 {
        self.get_config(category)
            .map(|c| c.error_probability)
            .or(self.default_error)
            .unwrap_or(0.0)
    }
}

impl NoiseChannel for CategoryBasedChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        // We respond to AfterGate events, but we need context to check category.
        // Return true here and do the check in apply().
        matches!(event, NoiseEvent::AfterGate { .. })
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

        // Get gate category from definitions
        let Some(gate_id) = gate_id else {
            return NoiseResponse::None;
        };

        let Some(category) = ctx.category(*gate_id) else {
            return NoiseResponse::None; // No definitions or unknown gate
        };

        let error_prob = self.error_for_category(category);
        if error_prob <= 0.0 {
            return NoiseResponse::None;
        }

        let default_weights = PauliWeights::uniform();
        let pauli_weights = self
            .get_config(category)
            .map_or(&default_weights, |c| &c.pauli_weights);

        let mut gates = SmallVec::new();

        // Fast path: check if any leakage exists at all
        let has_any_leakage = ctx.leaked_count() > 0;

        for &qubit in *qubits {
            // Skip leaked qubits
            if has_any_leakage && ctx.is_leaked(qubit) {
                continue;
            }

            // Apply error with probability
            if rng.random::<f64>() < error_prob {
                let pauli = pauli_weights.sample(rng.random::<f64>());
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
        "CategoryBasedChannel"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

/// Convert category to array index.
fn category_to_index(category: GateCategory) -> usize {
    match category {
        GateCategory::SingleQubitUnitary => 0,
        GateCategory::TwoQubitUnitary => 1,
        GateCategory::MultiQubitUnitary => 2,
        GateCategory::Preparation => 3,
        GateCategory::Measurement => 4,
        GateCategory::Idle => 5,
        GateCategory::QubitManagement => 6,
        GateCategory::Custom(_) => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::GateType;
    use crate::extensible::{GateDefinitions, gates};
    use pecos_core::QubitId;

    #[test]
    fn test_category_channel_single_qubit() {
        let channel = CategoryBasedChannel::new()
            .with_category(GateCategory::SingleQubitUnitary, 1.0)
            .with_category(GateCategory::TwoQubitUnitary, 0.0);

        let gates_def = GateDefinitions::new();
        let mut ctx = NoiseContext::new();
        ctx.set_gate_definitions(gates_def);

        let qubits = [QubitId(0)];
        let angles = [];
        let mut rng = PecosRng::seed_from_u64(42);

        // H gate (single-qubit) should get noise
        let h_event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: Some(gates::H),
        };

        let response = channel.apply(&h_event, &mut ctx, &mut rng);
        assert!(matches!(response, NoiseResponse::InjectGates(_)));
    }

    #[test]
    fn test_category_channel_two_qubit() {
        let channel = CategoryBasedChannel::new()
            .with_category(GateCategory::SingleQubitUnitary, 0.0)
            .with_category(GateCategory::TwoQubitUnitary, 1.0);

        let gates_def = GateDefinitions::new();
        let mut ctx = NoiseContext::new();
        ctx.set_gate_definitions(gates_def);

        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];
        let mut rng = PecosRng::seed_from_u64(42);

        // CX gate (two-qubit) should get noise
        let cx_event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &angles,
            gate_id: Some(gates::CX),
        };

        let response = channel.apply(&cx_event, &mut ctx, &mut rng);
        assert!(matches!(response, NoiseResponse::InjectGates(_)));
    }

    #[test]
    fn test_category_channel_custom_gate() {
        let mut gates_def = GateDefinitions::new();

        // Register a custom two-qubit gate
        let my_gate = gates_def.register(
            crate::extensible::GateSpec::new("MyGate")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );

        let channel = CategoryBasedChannel::new().with_category(GateCategory::TwoQubitUnitary, 1.0);

        let mut ctx = NoiseContext::new();
        ctx.set_gate_definitions(gates_def);

        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];
        let mut rng = PecosRng::seed_from_u64(42);

        // Custom gate should get same treatment as core two-qubit gates
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::I, // Placeholder - custom gates may not have GateType
            qubits: &qubits,
            angles: &angles,
            gate_id: Some(my_gate),
        };

        let response = channel.apply(&event, &mut ctx, &mut rng);
        assert!(matches!(response, NoiseResponse::InjectGates(_)));
    }

    #[test]
    fn test_category_channel_no_definitions() {
        let channel =
            CategoryBasedChannel::new().with_category(GateCategory::SingleQubitUnitary, 1.0);

        // No gate definitions set
        let mut ctx = NoiseContext::new();

        let qubits = [QubitId(0)];
        let angles = [];
        let mut rng = PecosRng::seed_from_u64(42);

        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: Some(gates::H),
        };

        // Without definitions, should return None (can't determine category)
        let response = channel.apply(&event, &mut ctx, &mut rng);
        assert!(response.is_none());
    }
}
