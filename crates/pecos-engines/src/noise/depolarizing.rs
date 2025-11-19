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

use crate::Gate;
use crate::byte_message::{ByteMessage, ByteMessageBuilder, GateType};
use crate::engine_system::{ControlEngine, EngineStage};
use crate::noise::{NoiseModel, NoiseRng, NoiseUtils, ProbabilityValidator, RngManageable};
use log::trace;
use pecos_core::errors::PecosError;
use rand_chacha::ChaCha8Rng;
use std::any::Any;

/// Implements depolarizing channel noise for quantum simulations
///
/// This model applies different error probabilities to various quantum operations:
/// - `p_prep`: Preparation error probability
/// - `p_meas`: Measurement error probability
/// - `p1`: Single-qubit gate error probability
/// - `p2`: Two-qubit gate error probability
///
/// # Usage
///
/// ```rust
/// use pecos_engines::noise::DepolarizingNoiseModel;
/// use pecos_engines::noise::{NoiseModel, RngManageable};
///
/// // Create with direct constructor
/// let mut noise_model = DepolarizingNoiseModel::new(0.01, 0.02, 0.03, 0.04);
/// noise_model.set_seed(42).unwrap(); // For reproducibility
///
/// // Or use the builder pattern
/// let noise_model = DepolarizingNoiseModel::builder()
///     .with_prep_probability(0.01)
///     .with_meas_probability(0.02)
///     .with_single_qubit_probability(0.03)
///     .with_two_qubit_probability(0.04)
///     .with_seed(42)
///     .build();
///
/// // Or use uniform probability
/// let noise_model = DepolarizingNoiseModel::builder()
///     .with_uniform_probability(0.01)
///     .build();
/// ```
#[derive(Clone)]
pub struct DepolarizingNoiseModel {
    /// Probability of applying an error during preparation
    p_prep: f64,
    /// Probability of applying an error during measurement
    p_meas: f64,
    /// Probability of applying an error after single-qubit gates
    p1: f64,
    /// Probability of applying an error after two-qubit gates
    p2: f64,
    /// Random number generator
    rng: NoiseRng<ChaCha8Rng>,
}

impl ProbabilityValidator for DepolarizingNoiseModel {}

impl DepolarizingNoiseModel {
    /// Create a new depolarizing noise model with the given probabilities
    #[must_use]
    pub fn new(p_prep: f64, p_meas: f64, p1: f64, p2: f64) -> Self {
        // Validate all probabilities
        Self::validate_probability(p_prep);
        Self::validate_probability(p_meas);
        Self::validate_probability(p1);
        Self::validate_probability(p2);

        Self {
            p_prep,
            p_meas,
            p1,
            p2,
            rng: NoiseRng::default(),
        }
    }

    /// Create a new noise model with uniform probability for all error types
    #[must_use]
    pub fn new_uniform(probability: f64) -> Self {
        Self::new(probability, probability, probability, probability)
    }

    /// Create a new builder for the depolarizing noise model
    #[must_use]
    pub fn builder() -> DepolarizingNoiseModelBuilder {
        DepolarizingNoiseModelBuilder::new()
    }

    /// Set all probabilities of error
    pub fn set_probabilities(&mut self, p_prep: f64, p_meas: f64, p1: f64, p2: f64) {
        Self::validate_probability(p_prep);
        Self::validate_probability(p_meas);
        Self::validate_probability(p1);
        Self::validate_probability(p2);

        self.p_prep = p_prep;
        self.p_meas = p_meas;
        self.p1 = p1;
        self.p2 = p2;
    }

    /// Set a uniform probability for all error types
    pub fn set_uniform_probability(&mut self, probability: f64) {
        self.set_probabilities(probability, probability, probability, probability);
    }

    /// Get the current error probabilities
    #[must_use]
    pub fn probabilities(&self) -> (f64, f64, f64, f64) {
        (self.p_prep, self.p_meas, self.p1, self.p2)
    }

    /// Apply noise to a list of quantum gates
    fn apply_noise_to_gates(&mut self, gates: &[Gate]) -> ByteMessage {
        let mut builder = NoiseUtils::create_quantum_builder();

        for gate in gates {
            match gate.gate_type {
                GateType::X
                | GateType::Y
                | GateType::Z
                | GateType::SZ
                | GateType::SZdg
                | GateType::H
                | GateType::T
                | GateType::Tdg
                | GateType::RX
                | GateType::RY
                | GateType::R1XY
                | GateType::U => {
                    NoiseUtils::add_gate_to_builder(&mut builder, gate);
                    trace!("Applying single-qubit gate with possible fault");
                    self.apply_sq_faults(&mut builder, gate);
                }
                GateType::CX | GateType::RZZ | GateType::SZZ | GateType::SZZdg => {
                    NoiseUtils::add_gate_to_builder(&mut builder, gate);
                    trace!("Applying two-qubit gate with possible fault");
                    self.apply_tq_faults(&mut builder, gate);
                }
                GateType::RZ => {
                    NoiseUtils::add_gate_to_builder(&mut builder, gate);
                }
                GateType::Measure | GateType::MeasureLeaked => {
                    trace!("Applying measurement with possible fault");
                    self.apply_meas_faults(&mut builder, gate);
                    NoiseUtils::add_gate_to_builder(&mut builder, gate);
                }
                GateType::Prep => {
                    NoiseUtils::add_gate_to_builder(&mut builder, gate);
                    trace!("Applying preparation with possible fault");
                    self.apply_prep_faults(&mut builder, gate);
                }
                GateType::I
                | GateType::Idle
                | GateType::MeasCrosstalkLocalPayload
                | GateType::MeasCrosstalkGlobalPayload => {
                    // Just pass through with no added noise
                    // builder.add_quantum_gate(gate);
                }
            }
        }

        builder.build()
    }

    fn apply_prep_faults(&mut self, builder: &mut ByteMessageBuilder, gate: &Gate) {
        if self.rng.occurs(self.p_prep) {
            trace!("Applying prep fault on qubits {:?}", gate.qubits);
            NoiseUtils::apply_x(builder, *gate.qubits[0]);
        }
    }

    fn apply_meas_faults(&mut self, builder: &mut ByteMessageBuilder, gate: &Gate) {
        if self.rng.occurs(self.p_meas) {
            trace!("Applying meas fault on qubits {:?}", gate.qubits);
            NoiseUtils::apply_x(builder, *gate.qubits[0]);
        }
    }

    fn apply_sq_faults(&mut self, builder: &mut ByteMessageBuilder, gate: &Gate) {
        if self.rng.occurs(self.p1) {
            let fault_type = self.rng.random_int(0..3);
            let qubit = gate.qubits[0];

            match fault_type {
                0 => {
                    trace!("Applying X fault on qubit {qubit}");
                    NoiseUtils::apply_x(builder, *qubit);
                }
                1 => {
                    trace!("Applying Y fault on qubit {qubit}");
                    NoiseUtils::apply_y(builder, *qubit);
                }
                _ => {
                    trace!("Applying Z fault on qubit {qubit}");
                    NoiseUtils::apply_z(builder, *qubit);
                }
            }
        }
    }

    fn apply_tq_faults(&mut self, builder: &mut ByteMessageBuilder, gate: &Gate) {
        if self.rng.occurs(self.p2) {
            let fault_type = self.rng.random_int(0..15);
            let qubit0 = gate.qubits[0];
            let qubit1 = gate.qubits[1];

            match fault_type {
                // IX
                0 => {
                    trace!("Applying IX fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_x(builder, *qubit1);
                }
                // IY
                1 => {
                    trace!("Applying IY fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_y(builder, *qubit1);
                }
                // IZ
                2 => {
                    trace!("Applying IZ fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_z(builder, *qubit1);
                }
                // XI
                3 => {
                    trace!("Applying XI fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_x(builder, *qubit0);
                }
                // XX
                4 => {
                    trace!("Applying XX fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_x(builder, *qubit0);
                    NoiseUtils::apply_x(builder, *qubit1);
                }
                // XY
                5 => {
                    trace!("Applying XY fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_x(builder, *qubit0);
                    NoiseUtils::apply_y(builder, *qubit1);
                }
                // XZ
                6 => {
                    trace!("Applying XZ fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_x(builder, *qubit0);
                    NoiseUtils::apply_z(builder, *qubit1);
                }
                // YI
                7 => {
                    trace!("Applying YI fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_y(builder, *qubit0);
                }
                // YX
                8 => {
                    trace!("Applying YX fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_y(builder, *qubit0);
                    NoiseUtils::apply_x(builder, *qubit1);
                }
                // YY
                9 => {
                    trace!("Applying YY fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_y(builder, *qubit0);
                    NoiseUtils::apply_y(builder, *qubit1);
                }
                // YZ
                10 => {
                    trace!("Applying YZ fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_y(builder, *qubit0);
                    NoiseUtils::apply_z(builder, *qubit1);
                }
                // ZI
                11 => {
                    trace!("Applying ZI fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_z(builder, *qubit0);
                }
                // ZX
                12 => {
                    trace!("Applying ZX fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_z(builder, *qubit0);
                    NoiseUtils::apply_x(builder, *qubit1);
                }
                // ZY
                13 => {
                    trace!("Applying ZY fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_z(builder, *qubit0);
                    NoiseUtils::apply_y(builder, *qubit1);
                }
                // ZZ
                _ => {
                    trace!("Applying ZZ fault on qubits {:?}", gate.qubits);
                    NoiseUtils::apply_z(builder, *qubit0);
                    NoiseUtils::apply_z(builder, *qubit1);
                }
            }
        }
    }
}

impl NoiseModel for DepolarizingNoiseModel {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl RngManageable for DepolarizingNoiseModel {
    type Rng = ChaCha8Rng;

    fn set_rng(&mut self, rng: ChaCha8Rng) -> Result<(), PecosError> {
        self.rng = NoiseRng::new(rng);
        Ok(())
    }

    fn rng(&self) -> &Self::Rng {
        self.rng.inner()
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.rng.inner_mut()
    }
}

/// Builder for creating depolarizing noise models
#[derive(Debug, Clone)]
pub struct DepolarizingNoiseModelBuilder {
    p_prep: Option<f64>,
    p_meas: Option<f64>,
    p1: Option<f64>,
    p2: Option<f64>,
    seed: Option<u64>,
}

impl Default for DepolarizingNoiseModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DepolarizingNoiseModelBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            p_prep: None,
            p_meas: None,
            p1: None,
            p2: None,
            seed: None,
        }
    }

    /// Set the same probability for all error types
    ///
    /// This is a convenience method to set all probabilities to the same value.
    ///
    /// # Arguments
    /// * `probability` - The probability value to set for all error types
    #[must_use]
    pub fn with_uniform_probability(mut self, probability: f64) -> Self {
        self.p_prep = Some(probability);
        self.p_meas = Some(probability);
        self.p1 = Some(probability);
        self.p2 = Some(probability);
        self
    }

    /// Set the probability of error during preparation
    #[must_use]
    pub fn with_prep_probability(mut self, probability: f64) -> Self {
        self.p_prep = Some(probability);
        self
    }

    /// Set the probability of error during measurement
    #[must_use]
    pub fn with_meas_probability(mut self, probability: f64) -> Self {
        self.p_meas = Some(probability);
        self
    }

    /// Set the probability of error after single-qubit gates
    #[must_use]
    pub fn with_p1_probability(mut self, probability: f64) -> Self {
        self.p1 = Some(probability);
        self
    }

    /// Set the probability of error after single-qubit gates
    ///
    /// This is an alias for `with_p1_probability` for API consistency.
    #[must_use]
    pub fn with_single_qubit_probability(self, probability: f64) -> Self {
        self.with_p1_probability(probability)
    }

    /// Set the probability of error after two-qubit gates
    #[must_use]
    pub fn with_p2_probability(mut self, probability: f64) -> Self {
        self.p2 = Some(probability);
        self
    }

    /// Set the probability of error after two-qubit gates
    ///
    /// This is an alias for `with_p2_probability` for API consistency.
    #[must_use]
    pub fn with_two_qubit_probability(self, probability: f64) -> Self {
        self.with_p2_probability(probability)
    }

    /// Set the seed for the random number generator
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the depolarizing noise model
    ///
    /// # Returns
    /// A `DepolarizingNoiseModel` instance
    ///
    /// # Panics
    /// Panics if any probabilities are not between 0 and 1.
    #[must_use]
    pub fn build(self) -> DepolarizingNoiseModel {
        let p_prep = self.p_prep.expect("Preparation probability must be set");
        let p_meas = self.p_meas.expect("Measurement probability must be set");
        let p1 = self.p1.expect("Single-qubit probability must be set");
        let p2 = self.p2.expect("Two-qubit probability must be set");

        // Create the noise model
        let mut noise = DepolarizingNoiseModel::new(p_prep, p_meas, p1, p2);

        // Set the seed if provided
        if let Some(seed) = self.seed {
            // Use RngManageable::set_seed directly
            noise.set_seed(seed).expect("Failed to set seed");
        }

        noise
    }
}

impl crate::noise::IntoNoiseModel for DepolarizingNoiseModelBuilder {
    fn into_noise_model(self) -> Box<dyn crate::noise::NoiseModel> {
        Box::new(self.build())
    }
}

impl ControlEngine for DepolarizingNoiseModel {
    type Input = ByteMessage;
    type Output = ByteMessage;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(
        &mut self,
        input: Self::Input,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        // For quantum operations, apply gate noise
        trace!("DepolarizingNoise::start - applying noise to quantum operations");

        // Parse the input as quantum operations
        let gates: Vec<crate::Gate> = input
            .quantum_ops()
            .map_err(|e| PecosError::Input(format!("Failed to parse quantum operations: {e}")))?;

        // Apply noise to the gates
        let noisy_gates = self.apply_noise_to_gates(&gates);

        // Return the noisy operations
        Ok(EngineStage::NeedsProcessing(noisy_gates))
    }

    fn continue_processing(
        &mut self,
        result: Self::EngineOutput,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        // This noise model doesn't directly modify measurement results, just pass through
        trace!("DepolarizingNoise::continue_processing - passing through measurement results");
        Ok(EngineStage::Complete(result))
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // No state to reset
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_system::{ControlEngine, EngineStage};

    #[test]
    fn test_probabilities_getter_and_setter() {
        // Create a noise model with initial probabilities
        let mut noise = DepolarizingNoiseModel::new(0.01, 0.02, 0.03, 0.04);

        // Check initial probabilities
        let (p_prep, p_meas, p1, p2) = noise.probabilities();
        assert!((p_prep - 0.01).abs() < f64::EPSILON);
        assert!((p_meas - 0.02).abs() < f64::EPSILON);
        assert!((p1 - 0.03).abs() < f64::EPSILON);
        assert!((p2 - 0.04).abs() < f64::EPSILON);

        // Update probabilities and check they were updated
        noise.set_probabilities(0.05, 0.06, 0.07, 0.08);
        let (p_prep, p_meas, p1, p2) = noise.probabilities();
        assert!((p_prep - 0.05).abs() < f64::EPSILON);
        assert!((p_meas - 0.06).abs() < f64::EPSILON);
        assert!((p1 - 0.07).abs() < f64::EPSILON);
        assert!((p2 - 0.08).abs() < f64::EPSILON);
    }

    #[test]
    fn test_uniform_probability() {
        // Test the uniform probability constructor
        let noise = DepolarizingNoiseModel::new_uniform(0.05);
        let (p_prep, p_meas, p1, p2) = noise.probabilities();
        assert!((p_prep - 0.05).abs() < f64::EPSILON);
        assert!((p_meas - 0.05).abs() < f64::EPSILON);
        assert!((p1 - 0.05).abs() < f64::EPSILON);
        assert!((p2 - 0.05).abs() < f64::EPSILON);

        // Test the uniform probability setter
        let mut noise = DepolarizingNoiseModel::new(0.01, 0.02, 0.03, 0.04);
        noise.set_uniform_probability(0.07);
        let (p_prep, p_meas, p1, p2) = noise.probabilities();
        assert!((p_prep - 0.07).abs() < f64::EPSILON);
        assert!((p_meas - 0.07).abs() < f64::EPSILON);
        assert!((p1 - 0.07).abs() < f64::EPSILON);
        assert!((p2 - 0.07).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "Probability must be between 0.0 and 1.0")]
    fn test_invalid_probability_panics() {
        let mut noise = DepolarizingNoiseModel::new(0.1, 0.2, 0.3, 0.4);
        noise.set_probabilities(0.1, 0.2, 1.1, 0.4); // Should panic
    }

    #[test]
    fn test_builder() {
        // Create a noise model with the builder
        let mut noise = DepolarizingNoiseModel::builder()
            .with_prep_probability(0.1)
            .with_meas_probability(0.2)
            .with_p1_probability(0.3)
            .with_p2_probability(0.4)
            .build();

        // Create a direct instance with the same probabilities
        let mut direct_noise = DepolarizingNoiseModel::new(0.1, 0.2, 0.3, 0.4);

        // Create a simple message for testing
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.add_x(&[0]);
        let input = builder.build();

        // Process using the ControlEngine API instead of the old apply_noise method
        let result1 = noise
            .start(input.clone())
            .expect("Builder-created noise model failed");
        let result2 = direct_noise
            .start(input)
            .expect("Directly created noise model failed");

        // Verify we got a valid result that needs processing
        match result1 {
            EngineStage::NeedsProcessing(_) => (),
            EngineStage::Complete(_) => panic!("Expected NeedsProcessing stage"),
        }

        match result2 {
            EngineStage::NeedsProcessing(_) => (),
            EngineStage::Complete(_) => panic!("Expected NeedsProcessing stage"),
        }
    }

    #[test]
    fn test_builder_with_uniform_probability() {
        // Create a noise model with the builder using uniform probability
        let noise = DepolarizingNoiseModel::builder()
            .with_uniform_probability(0.05)
            .build();

        // Create a direct instance with the same uniform probability
        let direct_noise = DepolarizingNoiseModel::new_uniform(0.05);

        // Check that probabilities match
        let (p_prep1, p_meas1, p1_1, p2_1) = direct_noise.probabilities();

        // Get the boxed noise model's probabilities using any_ref downcast
        let noise_ref = noise
            .as_any()
            .downcast_ref::<DepolarizingNoiseModel>()
            .unwrap();
        let (p_prep2, p_meas2, p1_2, p2_2) = noise_ref.probabilities();

        assert!((p_prep1 - p_prep2).abs() < f64::EPSILON);
        assert!((p_meas1 - p_meas2).abs() < f64::EPSILON);
        assert!((p1_1 - p1_2).abs() < f64::EPSILON);
        assert!((p2_1 - p2_2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_as_any_methods() {
        // Create a noise model
        let mut noise = DepolarizingNoiseModel::new(0.1, 0.2, 0.3, 0.4);

        // Test as_any for type checking
        assert!(noise.as_any().is::<DepolarizingNoiseModel>());

        // Test as_any_mut for downcasting and modifying
        let downcast_noise = noise
            .as_any_mut()
            .downcast_mut::<DepolarizingNoiseModel>()
            .unwrap();
        downcast_noise.set_probabilities(0.5, 0.5, 0.5, 0.5);

        let (p_prep, p_meas, p1, p2) = noise.probabilities();
        assert!((p_prep - 0.5).abs() < f64::EPSILON);
        assert!((p_meas - 0.5).abs() < f64::EPSILON);
        assert!((p1 - 0.5).abs() < f64::EPSILON);
        assert!((p2 - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_builder_with_probability() {
        // Create a noise model with the builder
        let mut noise = DepolarizingNoiseModel::builder()
            .with_prep_probability(0.01)
            .with_meas_probability(0.02)
            .with_p1_probability(0.03)
            .with_p2_probability(0.04)
            .build();

        // Create a direct instance with the same probabilities
        let mut direct_noise = DepolarizingNoiseModel::new(0.01, 0.02, 0.03, 0.04);

        // Create a simple quantum operations message for testing
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.add_x(&[0]);
        let input = builder.build();

        // Process using the ControlEngine API instead of the old apply_noise method
        let result1 = noise
            .start(input.clone())
            .expect("Builder-created noise model failed");
        let result2 = direct_noise
            .start(input)
            .expect("Directly created noise model failed");

        // Verify we got a valid result that needs processing
        match result1 {
            EngineStage::NeedsProcessing(_) => (),
            EngineStage::Complete(_) => panic!("Expected NeedsProcessing stage"),
        }

        match result2 {
            EngineStage::NeedsProcessing(_) => (),
            EngineStage::Complete(_) => panic!("Expected NeedsProcessing stage"),
        }
    }
}
