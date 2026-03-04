// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Utility functions for noise models.
//!
//! The main responsibility of this file is to generate random operations
//! and convert them to quantum gates (suitable for adding to the `ByteMessage`).

#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_panics_doc)]

use crate::Gate;
use crate::byte_message::{ByteMessage, ByteMessageBuilder};

/// Helper trait for validating probability values
pub trait ProbabilityValidator {
    /// Validate that a probability is between 0.0 and 1.0
    ///
    /// # Arguments
    /// * `probability` - The probability value to validate
    ///
    /// # Panics
    /// Panics if the probability is not between 0.0 and 1.0
    fn validate_probability(probability: f64) {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability must be between 0.0 and 1.0"
        );
    }

    /// Validate a named probability
    ///
    /// # Arguments
    /// * `probability` - The probability value to validate
    /// * `name` - Name of the probability for error message
    ///
    /// # Panics
    /// Panics if the probability is not between 0.0 and 1.0
    fn validate_named_probability(probability: f64, name: &str) {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability {name} must be between 0.0 and 1.0, but was {probability}"
        );
    }
}

/// Helper functions for working with quantum gates and noise
pub struct NoiseUtils;

/// Result of a single-qubit operation sampling that may include leakage
#[derive(Debug)]
pub struct SingleQubitNoiseResult {
    /// Optional quantum gate to apply (None if only leakage)
    pub gate: Option<Gate>,
    /// Whether the qubit leaked
    pub qubit_leaked: bool,
}

impl SingleQubitNoiseResult {
    /// Creates a new `SingleQubitNoiseResult` with a Pauli gate and no leakage
    #[must_use]
    pub fn with_gate(gate: Gate) -> Self {
        Self {
            gate: Some(gate),
            qubit_leaked: false,
        }
    }

    /// Creates a new `SingleQubitNoiseResult` with leakage and no gate
    #[must_use]
    pub fn with_leakage() -> Self {
        Self {
            gate: None,
            qubit_leaked: true,
        }
    }

    /// Whether this result includes leakage
    #[must_use]
    pub fn has_leakage(&self) -> bool {
        self.qubit_leaked
    }

    #[must_use]
    pub fn has_leakages(&self) -> Vec<bool> {
        vec![self.qubit_leaked]
    }
}

/// Result of a two-qubit operation sampling that may include leakage
#[derive(Debug)]
pub struct TwoQubitNoiseResult {
    /// Quantum gates to apply (None if operations are just leakage or identity)
    pub gates: Option<Vec<Gate>>,
    /// Whether the first qubit leaked
    pub qubit0_leaked: bool,
    /// Whether the second qubit leaked
    pub qubit1_leaked: bool,
}

impl TwoQubitNoiseResult {
    /// Creates a new `TwoQubitNoiseResult` indicating leakage on a qubit
    #[must_use]
    pub fn with_leakage(
        qubit0_leaked: bool,
        qubit1_leaked: bool,
        gates: Option<Vec<Gate>>,
    ) -> Self {
        Self {
            gates,
            qubit0_leaked,
            qubit1_leaked,
        }
    }

    /// Creates a new `TwoQubitNoiseResult` with just gates and no leakage
    #[must_use]
    pub fn with_gates(gates: Vec<Gate>) -> Self {
        Self {
            gates: Some(gates),
            qubit0_leaked: false,
            qubit1_leaked: false,
        }
    }

    /// Whether any qubit leaked
    #[must_use]
    pub fn has_leakage(&self) -> bool {
        self.qubit0_leaked || self.qubit1_leaked
    }

    #[must_use]
    pub fn has_leakages(&self) -> Vec<bool> {
        vec![self.qubit0_leaked, self.qubit1_leaked]
    }

    /// Whether the first qubit leaked
    #[must_use]
    pub fn has_qubit0_leakage(&self) -> bool {
        self.qubit0_leaked
    }

    /// Whether the second qubit leaked
    #[must_use]
    pub fn has_qubit1_leakage(&self) -> bool {
        self.qubit1_leaked
    }
}

impl NoiseUtils {
    /// Add a gate to a builder based on its type
    ///
    /// This is a utility function to add a gate to a `ByteMessageBuilder`
    /// based on its type, handling all the common gate types.
    ///
    /// # Arguments
    /// * `builder` - The `ByteMessageBuilder` to add the gate to
    /// * `gate` - The gate to add
    ///
    /// # Panics
    /// Panics if:
    /// - `gate` is `None` when processing a measurement gate
    /// - The gate type is invalid or has insufficient parameters/qubits for the operation
    pub fn add_gate_to_builder(builder: &mut ByteMessageBuilder, gate: &Gate) {
        use crate::byte_message::GateType;

        match gate.gate_type {
            // Single-qubit gates that operate directly on qubit lists
            GateType::X => {
                let qubits_usize: Vec<usize> = gate.qubits.iter().map(|q| **q).collect();
                builder.add_x(&qubits_usize);
            }
            GateType::Y => {
                let qubits_usize: Vec<usize> = gate.qubits.iter().map(|q| **q).collect();
                builder.add_y(&qubits_usize);
            }
            GateType::Z => {
                let qubits_usize: Vec<usize> = gate.qubits.iter().map(|q| **q).collect();
                builder.add_z(&qubits_usize);
            }
            GateType::H => {
                let qubits_usize: Vec<usize> = gate.qubits.iter().map(|q| **q).collect();
                builder.add_h(&qubits_usize);
            }
            GateType::Prep => {
                let qubits_usize: Vec<usize> = gate.qubits.iter().map(|q| **q).collect();
                builder.add_prep(&qubits_usize);
            }

            // Two-qubit gates that need qubit validation
            GateType::CX if gate.qubits.len() >= 2 => {
                builder.add_cx(&[*gate.qubits[0]], &[*gate.qubits[1]]);
            }
            GateType::SZZ if gate.qubits.len() >= 2 => {
                builder.add_szz(&[*gate.qubits[0]], &[*gate.qubits[1]]);
            }
            GateType::SZZdg if gate.qubits.len() >= 2 => {
                builder.add_szzdg(&[*gate.qubits[0]], &[*gate.qubits[1]]);
            }

            // Rotation gates - angles are now stored in gate.angles field
            GateType::RX if !gate.angles.is_empty() => {
                let qubits_usize: Vec<usize> = gate.qubits.iter().map(|q| **q).collect();
                builder.add_rx(gate.angles[0], &qubits_usize);
            }
            GateType::RY if !gate.angles.is_empty() => {
                let qubits_usize: Vec<usize> = gate.qubits.iter().map(|q| **q).collect();
                builder.add_ry(gate.angles[0], &qubits_usize);
            }
            GateType::RZ if !gate.angles.is_empty() => {
                let qubits_usize: Vec<usize> = gate.qubits.iter().map(|q| **q).collect();
                builder.add_rz(gate.angles[0], &qubits_usize);
            }
            GateType::RZZ if gate.qubits.len() >= 2 && !gate.angles.is_empty() => {
                builder.add_rzz(gate.angles[0], &[*gate.qubits[0]], &[*gate.qubits[1]]);
            }
            GateType::R1XY if gate.angles.len() >= 2 => {
                let qubits_usize: Vec<usize> = gate.qubits.iter().map(|q| **q).collect();
                builder.add_r1xy(gate.angles[0], gate.angles[1], &qubits_usize);
            }
            GateType::U if gate.angles.len() >= 3 => {
                let qubits_usize: Vec<usize> = gate.qubits.iter().map(|q| **q).collect();
                builder.add_u(
                    gate.angles[0],
                    gate.angles[1],
                    gate.angles[2],
                    &qubits_usize,
                );
            }

            // Measurement gates
            GateType::Measure if !gate.qubits.is_empty() => {
                let qubits_usize: Vec<usize> = gate.qubits.iter().map(|q| **q).collect();
                builder.add_measurements(&qubits_usize);
            }

            // Idle gates need special handling for qubit lists
            GateType::Idle if !gate.params.is_empty() => {
                // Use gate params for idle time
                let qubits_usize: Vec<usize> = gate.qubits.iter().map(|q| **q).collect();
                builder.add_idle(gate.params[0], &qubits_usize);
            }

            // Custom is a placeholder (actual gate name is in metadata) -- skip.
            GateType::Custom => {}

            // All other gates: use generic serialization (gate type + qubits + angles/params).
            _ => {
                builder.add_gate_command(gate);
            }
        }
    }

    /// Check if a message contains measurement results
    ///
    /// # Arguments
    /// * `message` - The `ByteMessage` to check
    ///
    /// # Returns
    /// true if the message contains measurement results, false otherwise
    #[must_use]
    pub fn has_measurements(message: &ByteMessage) -> bool {
        message.outcomes().is_ok_and(|m| !m.is_empty())
    }

    /// Creates a new `ByteMessageBuilder` for quantum operations
    ///
    /// # Returns
    /// A `ByteMessageBuilder` configured for quantum operations
    #[must_use]
    pub fn create_quantum_builder() -> ByteMessageBuilder {
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder
    }

    /// Creates a new `ByteMessage` from a list of gates
    ///
    /// # Arguments
    /// * `gates` - The gates to include in the message
    ///
    /// # Returns
    /// A `ByteMessage` containing the gates
    #[must_use]
    pub fn create_gate_message(gates: &[Gate]) -> ByteMessage {
        let mut builder = Self::create_quantum_builder();
        for gate in gates {
            Self::add_gate_to_builder(&mut builder, gate);
        }
        builder.build()
    }

    /// Applies X gate to a qubit via a builder
    ///
    /// # Arguments
    /// * `builder` - The `ByteMessageBuilder` to add the gate to
    /// * `qubit` - The qubit to apply the gate to
    pub fn apply_x(builder: &mut ByteMessageBuilder, qubit: usize) {
        builder.add_x(&[qubit]);
    }

    /// Applies Y gate to a qubit via a builder
    ///
    /// # Arguments
    /// * `builder` - The `ByteMessageBuilder` to add the gate to
    /// * `qubit` - The qubit to apply the gate to
    pub fn apply_y(builder: &mut ByteMessageBuilder, qubit: usize) {
        builder.add_y(&[qubit]);
    }

    /// Applies Z gate to a qubit via a builder
    ///
    /// # Arguments
    /// * `builder` - The `ByteMessageBuilder` to add the gate to
    /// * `qubit` - The qubit to apply the gate to
    pub fn apply_z(builder: &mut ByteMessageBuilder, qubit: usize) {
        builder.add_z(&[qubit]);
    }

    /// Apply a Pauli gate based on the Pauli string identifier
    ///
    /// # Arguments
    /// * `builder` - The `ByteMessageBuilder` to add the gate to
    /// * `pauli` - The Pauli gate identifier ("X", "Y", or "Z")
    /// * `qubit` - The qubit to apply the gate to
    ///
    /// # Returns
    /// `true` if a valid Pauli gate was applied, `false` otherwise
    pub fn apply_pauli(builder: &mut ByteMessageBuilder, pauli: &str, qubit: usize) -> bool {
        match pauli {
            "X" => {
                Self::apply_x(builder, qubit);
                true
            }
            "Y" => {
                Self::apply_y(builder, qubit);
                true
            }
            "Z" => {
                Self::apply_z(builder, qubit);
                true
            }
            _ => false,
        }
    }

    /// Create a Pauli gate based on the Pauli string identifier
    ///
    /// # Arguments
    /// * `pauli` - The Pauli gate identifier ("X", "Y", or "Z")
    /// * `qubit` - The qubit to apply the gate to
    ///
    /// # Returns
    /// A `Result` containing either the created `GateCommand` or an error if an invalid Pauli is provided
    ///
    /// # Errors
    /// Returns an error if the pauli string is not one of "X", "Y", or "Z"
    pub fn create_pauli_gate(pauli: &str, qubit: usize) -> Result<Gate, String> {
        match pauli {
            "X" => Ok(Gate::x(&[qubit])),
            "Y" => Ok(Gate::y(&[qubit])),
            "Z" => Ok(Gate::z(&[qubit])),
            _ => Err(format!("Invalid Pauli operator: {pauli}")),
        }
    }

    /// Prepares a qubit in the |0⟩ state via a builder
    ///
    /// # Arguments
    /// * `builder` - The `ByteMessageBuilder` to add the gate to
    /// * `qubit` - The qubit to prepare
    pub fn apply_prep_0(builder: &mut ByteMessageBuilder, qubit: usize) {
        builder.add_prep(&[qubit]);
    }

    /// Prepares a qubit in the |1⟩ state via a builder (applies prep followed by X)
    ///
    /// # Arguments
    /// * `builder` - The `ByteMessageBuilder` to add the gate to
    /// * `qubit` - The qubit to prepare
    pub fn apply_prep_1(builder: &mut ByteMessageBuilder, qubit: usize) {
        builder.add_prep(&[qubit]);
        builder.add_x(&[qubit]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::byte_message::GateType;
    use crate::noise::noise_rng::NoiseRng;
    use crate::noise::weighted_sampler::SingleQubitWeightedSampler;
    use pecos_rng::PecosRng;
    use std::collections::BTreeMap;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn test_noise_utils_create_quantum_builder() {
        let mut builder = NoiseUtils::create_quantum_builder();
        let message = builder.build();
        let result = message.quantum_ops();
        assert!(result.is_ok());
    }

    #[test]
    fn test_noise_utils_create_gate_message() {
        use crate::Gate;

        let gates = vec![Gate::x(&[0]), Gate::y(&[1])];

        let message = NoiseUtils::create_gate_message(&gates);
        let parsed_gates = message.quantum_ops().unwrap();
        assert_eq!(parsed_gates.len(), 2);
    }

    #[test]
    fn test_prep_functions() {
        use crate::byte_message::GateType;
        use pecos_core::QubitId;

        // Test preparation to |0⟩
        let mut builder = NoiseUtils::create_quantum_builder();
        NoiseUtils::apply_prep_0(&mut builder, 0);
        let message = builder.build();
        let parsed_gates = message.quantum_ops().unwrap();

        // Should have one Prep gate
        assert_eq!(parsed_gates.len(), 1);
        assert_eq!(parsed_gates[0].gate_type, GateType::Prep);
        assert_eq!(parsed_gates[0].qubits.as_slice(), &[QubitId(0)]);

        // Test preparation to |1⟩
        let mut builder = NoiseUtils::create_quantum_builder();
        NoiseUtils::apply_prep_1(&mut builder, 1);
        let message = builder.build();
        let parsed_gates = message.quantum_ops().unwrap();

        // Should have two gates: Prep followed by X
        assert_eq!(parsed_gates.len(), 2);
        assert_eq!(parsed_gates[0].gate_type, GateType::Prep);
        assert_eq!(parsed_gates[0].qubits.as_slice(), &[QubitId(1)]);
        assert_eq!(parsed_gates[1].gate_type, GateType::X);
        assert_eq!(parsed_gates[1].qubits.as_slice(), &[QubitId(1)]);
    }

    #[test]
    fn test_sample_paulis() {
        let mut rng = NoiseRng::<PecosRng>::with_seed(42);

        // Test with a valid model
        // Note: Weights must sum to exactly 1.0 to pass the strict normalization check
        let valid_model: BTreeMap<String, f64> = [
            ("X".to_string(), 0.5),
            ("Y".to_string(), 0.3),
            ("Z".to_string(), 0.2),
        ]
        .iter()
        .cloned()
        .collect();

        // Verify the sum is exactly 1.0
        let sum: f64 = valid_model.values().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Model weights should sum to 1.0, got {sum}"
        );

        // Create a SingleQubitWeightedSampler with the valid model
        // This sampler handles validation and normalization of weights
        let sampler = SingleQubitWeightedSampler::new(&valid_model);

        // Sample multiple times to ensure different outcomes
        let mut x_count = 0;
        let mut y_count = 0;
        let mut z_count = 0;

        for _ in 0..1000 {
            // Use the sampler to generate quantum gates based on the weighted probabilities
            let result = sampler.sample_gates(&mut rng, 0);

            // Only check gates (no leakage in this test)
            match result.gate {
                Some(gate) => match gate.gate_type {
                    GateType::X => x_count += 1,
                    GateType::Y => y_count += 1,
                    GateType::Z => z_count += 1,
                    _ => panic!("Unexpected gate type"),
                },
                None => panic!("Unexpected result: None"),
            }
        }

        // Given our weights, we expect roughly 50% X, 30% Y, 20% Z
        assert!(
            x_count > 400 && x_count < 600,
            "Expected ~500 X gates, got {x_count}"
        );
        assert!(
            y_count > 250 && y_count < 350,
            "Expected ~300 Y gates, got {y_count}"
        );
        assert!(
            z_count > 150 && z_count < 250,
            "Expected ~200 Z gates, got {z_count}"
        );

        // Test that invalid Pauli gate directly panics
        let result = catch_unwind(|| NoiseUtils::create_pauli_gate("INVALID", 0).unwrap());
        assert!(result.is_err(), "Should panic for invalid Pauli operator");

        // Test that empty model causes the sampler constructor to panic
        let empty_model: BTreeMap<String, f64> = BTreeMap::new();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = SingleQubitWeightedSampler::new(&empty_model);
        }));
        assert!(result.is_err(), "Should panic for empty model");

        // Test that model with invalid keys causes the sampler constructor to panic
        let invalid_keys: BTreeMap<String, f64> =
            [("X".to_string(), 0.5), ("INVALID".to_string(), 0.5)]
                .iter()
                .cloned()
                .collect();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = SingleQubitWeightedSampler::new(&invalid_keys);
        }));
        assert!(result.is_err(), "Should panic for invalid keys");
    }

    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::too_many_lines,
        clippy::no_effect_underscore_binding
    )]
    fn test_sample_sq_pauli_leakage_model() {
        use crate::byte_message::GateType;

        // Define constants at the beginning
        const SAMPLE_SIZE: usize = 10000;

        let mut rng = NoiseRng::<PecosRng>::with_seed(42);

        // Test with a valid model including leakage
        // Note: Weights must sum to exactly 1.0 to pass the strict normalization check
        let valid_model: BTreeMap<String, f64> = [
            ("X".to_string(), 0.4),
            ("Y".to_string(), 0.3),
            ("Z".to_string(), 0.2),
            ("L".to_string(), 0.1), // L represents leakage
        ]
        .iter()
        .cloned()
        .collect();

        // Verify the sum is exactly 1.0
        let sum: f64 = valid_model.values().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Model weights should sum to 1.0, got {sum}"
        );

        // Create a SingleQubitWeightedSampler with the valid model including leakage
        // This sampler handles both Pauli operations and leakage events
        let sampler = SingleQubitWeightedSampler::new(&valid_model);

        // Sample multiple times to test distribution
        let mut x_count = 0;
        let mut y_count = 0;
        let mut z_count = 0;
        let mut leakage_count = 0;

        for _ in 0..SAMPLE_SIZE {
            // Sample gates and check for both gate operations and leakage
            let result = sampler.sample_gates(&mut rng, 0);

            if result.qubit_leaked {
                leakage_count += 1;
            } else if let Some(gate) = result.gate {
                match gate.gate_type {
                    GateType::X => x_count += 1,
                    GateType::Y => y_count += 1,
                    GateType::Z => z_count += 1,
                    _ => panic!("Unexpected gate type"),
                }
            }
        }

        // Check that the distributions are roughly as expected
        let expected_x = (SAMPLE_SIZE as f64 * 0.4) as usize;
        let expected_y = (SAMPLE_SIZE as f64 * 0.3) as usize;
        let expected_z = (SAMPLE_SIZE as f64 * 0.2) as usize;
        let expected_l = (SAMPLE_SIZE as f64 * 0.1) as usize;

        let margin = SAMPLE_SIZE / 10; // Allow 10% margin for safety

        assert!(
            x_count.max(expected_x) - x_count.min(expected_x) < margin,
            "X count {x_count} deviates too much from expected {expected_x}"
        );
        assert!(
            y_count.max(expected_y) - y_count.min(expected_y) < margin,
            "Y count {y_count} deviates too much from expected {expected_y}"
        );
        assert!(
            z_count.max(expected_z) - z_count.min(expected_z) < margin,
            "Z count {z_count} deviates too much from expected {expected_z}"
        );
        assert!(
            leakage_count.max(expected_l) - leakage_count.min(expected_l) < margin,
            "Leakage count {leakage_count} deviates too much from expected {expected_l}"
        );

        // Verify the sum is correct
        assert_eq!(x_count + y_count + z_count + leakage_count, SAMPLE_SIZE);

        // Test error cases with catch_unwind
        let empty_model: BTreeMap<String, f64> = BTreeMap::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // This should trigger an "empty model" panic
            let _ = SingleQubitWeightedSampler::new(&empty_model);
        }));
        assert!(result.is_err(), "Empty model should cause panic");

        // Test invalid operation
        let invalid_model: BTreeMap<String, f64> = [
            ("X".to_string(), 0.3),
            ("INVALID".to_string(), 0.7), // Not a valid Pauli or L
        ]
        .iter()
        .cloned()
        .collect();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // This should trigger an "invalid operation" panic
            let _ = SingleQubitWeightedSampler::new(&invalid_model);
        }));
        assert!(result.is_err(), "Invalid operation should cause panic");
    }
}
