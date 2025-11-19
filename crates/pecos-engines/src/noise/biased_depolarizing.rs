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

/// Implements general noise model for quantum simulations, combining
/// depolarizing channel noise with biased measurement noise
///
/// This model applies different error probabilities to various quantum operations:
/// - `p_prep`: Preparation error probability
/// - `p_meas_0`: Probability of flipping a 0 measurement to 1
/// - `p_meas_1`: Probability of flipping a 1 measurement to 0
/// - `p1`: Single-qubit gate error probability
/// - `p2`: Two-qubit gate error probability
///
/// # Usage
///
/// ```rust
/// use pecos_engines::noise::BiasedDepolarizingNoiseModel;
/// use pecos_engines::noise::{NoiseModel, RngManageable};
///
/// // Create with direct constructor
/// let mut noise_model = BiasedDepolarizingNoiseModel::new(0.01, 0.02, 0.03, 0.04, 0.05);
/// noise_model.set_seed(42).unwrap(); // For reproducibility
///
/// // Or use the builder pattern
/// let noise_model = BiasedDepolarizingNoiseModel::builder()
///     .with_prep_probability(0.01)
///     .with_meas_0_probability(0.02)
///     .with_meas_1_probability(0.03)
///     .with_single_qubit_probability(0.04)
///     .with_two_qubit_probability(0.05)
///     .with_seed(42)
///     .build();
///
/// // Or use uniform probability
/// let noise_model = BiasedDepolarizingNoiseModel::builder()
///     .with_uniform_probability(0.01)
///     .build();
/// ```
#[derive(Clone)]
pub struct BiasedDepolarizingNoiseModel {
    /// Probability of applying an error during preparation
    p_prep: f64,
    /// Probability of flipping a 0 measurement to 1
    p_meas_0: f64,
    /// Probability of flipping a 1 measurement to 0
    p_meas_1: f64,
    /// Probability of applying an error after single-qubit gates
    p1: f64,
    /// Probability of applying an error after two-qubit gates
    p2: f64,
    /// Random number generator
    rng: NoiseRng<ChaCha8Rng>,
}

impl ProbabilityValidator for BiasedDepolarizingNoiseModel {}

impl BiasedDepolarizingNoiseModel {
    /// Create a new general noise model with the given probabilities
    #[must_use]
    pub fn new(p_prep: f64, p_meas_0: f64, p_meas_1: f64, p1: f64, p2: f64) -> Self {
        // Validate all probabilities
        Self::validate_probability(p_prep);
        Self::validate_probability(p_meas_0);
        Self::validate_probability(p_meas_1);
        Self::validate_probability(p1);
        Self::validate_probability(p2);

        Self {
            p_prep,
            p_meas_0,
            p_meas_1,
            p1,
            p2,
            rng: NoiseRng::default(),
        }
    }

    /// Create a new noise model with uniform probability for all error types
    #[must_use]
    pub fn new_uniform(probability: f64) -> Self {
        Self::new(
            probability,
            probability,
            probability,
            probability,
            probability,
        )
    }

    /// Create a new builder for the general noise model
    #[must_use]
    pub fn builder() -> BiasedDepolarizingNoiseModelBuilder {
        BiasedDepolarizingNoiseModelBuilder::new()
    }

    /// Set all probabilities of error
    pub fn set_probabilities(
        &mut self,
        p_prep: f64,
        p_meas_0: f64,
        p_meas_1: f64,
        p1: f64,
        p2: f64,
    ) {
        Self::validate_probability(p_prep);
        Self::validate_probability(p_meas_0);
        Self::validate_probability(p_meas_1);
        Self::validate_probability(p1);
        Self::validate_probability(p2);

        self.p_prep = p_prep;
        self.p_meas_0 = p_meas_0;
        self.p_meas_1 = p_meas_1;
        self.p1 = p1;
        self.p2 = p2;
    }

    /// Set a uniform probability for all error types
    pub fn set_uniform_probability(&mut self, probability: f64) {
        self.set_probabilities(
            probability,
            probability,
            probability,
            probability,
            probability,
        );
    }

    /// Get the current error probabilities
    #[must_use]
    pub fn probabilities(&self) -> (f64, f64, f64, f64, f64) {
        (self.p_prep, self.p_meas_0, self.p_meas_1, self.p1, self.p2)
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
                | GateType::RZ
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
                GateType::Measure | GateType::MeasureLeaked => {
                    trace!("Applying measurement. Will apply bias after engine returns results.");
                    // we apply biased measurement after the engine
                    // returns the results, rather than before measurement
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
                | GateType::MeasCrosstalkGlobalPayload => {}
            }
        }

        builder.build()
    }

    /// Apply bias to a single measurement result
    ///
    /// # Arguments
    /// * `result_id` - The result ID of the measurement
    /// * `outcome` - The outcome of the measurement (0 or 1)
    ///
    /// # Returns
    /// The potentially biased measurement outcome
    fn apply_bias_to_measurement(&mut self, result_id: u32, outcome: u32) -> (u32, u32) {
        // Generate a random number to determine if we should flip
        let should_flip = if outcome == 0 {
            // Flip from 0 to 1 with probability p_meas_0
            self.rng.occurs(self.p_meas_0)
        } else {
            // Flip from 1 to 0 with probability p_meas_1
            self.rng.occurs(self.p_meas_1)
        };

        if should_flip {
            // Flip the measurement outcome
            (result_id, 1 - outcome)
        } else {
            // Keep the original measurement
            (result_id, outcome)
        }
    }

    /// Apply bias to a `ByteMessage` containing measurement results
    ///
    /// # Arguments
    /// * `message` - The `ByteMessage` containing measurement results
    ///
    /// # Returns
    /// A new `ByteMessage` with biased measurement results
    ///
    /// # Errors
    /// Returns a `PecosError` if applying bias fails
    fn apply_bias_to_message(&mut self, message: ByteMessage) -> Result<ByteMessage, PecosError> {
        // Parse the message to extract the measurement results
        let outcomes = message.outcomes()?;

        // If the message doesn't contain measurements, return it unchanged
        if outcomes.is_empty() {
            return Ok(message);
        }

        // Apply bias to each measurement
        let biased_outcomes: Vec<u32> = outcomes
            .into_iter()
            .enumerate()
            .map(|(index, outcome)| {
                let index_u32 = u32::try_from(index).unwrap_or(u32::MAX);
                let (_biased_index, biased_outcome) =
                    self.apply_bias_to_measurement(index_u32, outcome);
                biased_outcome
            })
            .collect();

        // Create a new ByteMessage with the biased measurements using the builder
        let mut builder = ByteMessage::outcomes_builder();

        // Convert outcomes to usize for the builder
        let outcomes_usize: Vec<usize> = biased_outcomes
            .iter()
            .map(|&outcome| outcome as usize)
            .collect();
        builder.add_outcomes(&outcomes_usize);

        Ok(builder.build())
    }

    fn apply_prep_faults(&mut self, builder: &mut ByteMessageBuilder, gate: &Gate) {
        if self.rng.occurs(self.p_prep) {
            trace!("Applying prep fault on qubits {:?}", gate.qubits);
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

impl ControlEngine for BiasedDepolarizingNoiseModel {
    type Input = ByteMessage;
    type Output = ByteMessage;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(
        &mut self,
        input: Self::Input,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        // For quantum operations, apply gate noise
        trace!("BiasedDepolarizingNoise::start - applying noise to quantum operations");

        // Parse the input as quantum operations
        let gates: Vec<crate::Gate> = input.quantum_ops()?;

        // Apply noise to the gates
        let noisy_gates = self.apply_noise_to_gates(&gates);

        // Return the noisy operations
        Ok(EngineStage::NeedsProcessing(noisy_gates))
    }

    fn continue_processing(
        &mut self,
        result: Self::EngineOutput,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        // Apply biased measurement to measurement results
        trace!("BiasedDepolarizingNoise::continue_processing - applying biased measurement");
        let biased_result = self.apply_bias_to_message(result)?;
        Ok(EngineStage::Complete(biased_result))
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // No state to reset
        Ok(())
    }
}

impl NoiseModel for BiasedDepolarizingNoiseModel {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl RngManageable for BiasedDepolarizingNoiseModel {
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

/// Builder for creating biased depolarizing noise models
#[derive(Debug, Clone)]
pub struct BiasedDepolarizingNoiseModelBuilder {
    p_prep: Option<f64>,
    p_meas_0: Option<f64>,
    p_meas_1: Option<f64>,
    p1: Option<f64>,
    p2: Option<f64>,
    seed: Option<u64>,
}

impl Default for BiasedDepolarizingNoiseModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BiasedDepolarizingNoiseModelBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            p_prep: None,
            p_meas_0: None,
            p_meas_1: None,
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
        self.p_meas_0 = Some(probability);
        self.p_meas_1 = Some(probability);
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

    /// Set the probability of flipping 0 to 1 during measurement
    #[must_use]
    pub fn with_meas_0_probability(mut self, probability: f64) -> Self {
        self.p_meas_0 = Some(probability);
        self
    }

    /// Set the probability of flipping 1 to 0 during measurement
    #[must_use]
    pub fn with_meas_1_probability(mut self, probability: f64) -> Self {
        self.p_meas_1 = Some(probability);
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

    /// Build the general noise model
    ///
    /// # Returns
    /// A `BiasedDepolarizingNoiseModel` instance
    ///
    /// # Panics
    /// Panics if any probabilities are not set or are not between 0 and 1.
    #[must_use]
    pub fn build(self) -> BiasedDepolarizingNoiseModel {
        let p_prep = self.p_prep.expect("Preparation probability must be set");
        let p_meas_0 = self
            .p_meas_0
            .expect("Measurement 0->1 flip probability must be set");
        let p_meas_1 = self
            .p_meas_1
            .expect("Measurement 1->0 flip probability must be set");
        let p1 = self.p1.expect("Single-qubit probability must be set");
        let p2 = self.p2.expect("Two-qubit probability must be set");

        // Create the noise model
        let mut noise = BiasedDepolarizingNoiseModel::new(p_prep, p_meas_0, p_meas_1, p1, p2);

        // Set the seed if provided
        if let Some(seed) = self.seed {
            // Use RngManageable::set_seed directly
            noise.set_seed(seed).expect("Failed to set seed");
        }

        noise
    }
}

impl crate::noise::IntoNoiseModel for BiasedDepolarizingNoiseModelBuilder {
    fn into_noise_model(self) -> Box<dyn crate::noise::NoiseModel> {
        Box::new(self.build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probabilities_getter_and_setter() {
        // Create a noise model with initial probabilities
        let mut noise = BiasedDepolarizingNoiseModel::new(0.01, 0.02, 0.03, 0.04, 0.05);

        // Check initial probabilities
        let (p_prep, p_meas_0, p_meas_1, p1, p2) = noise.probabilities();
        assert!((p_prep - 0.01).abs() < f64::EPSILON);
        assert!((p_meas_0 - 0.02).abs() < f64::EPSILON);
        assert!((p_meas_1 - 0.03).abs() < f64::EPSILON);
        assert!((p1 - 0.04).abs() < f64::EPSILON);
        assert!((p2 - 0.05).abs() < f64::EPSILON);

        // Update probabilities and check they were updated
        noise.set_probabilities(0.05, 0.06, 0.07, 0.08, 0.09);
        let (p_prep, p_meas_0, p_meas_1, p1, p2) = noise.probabilities();
        assert!((p_prep - 0.05).abs() < f64::EPSILON);
        assert!((p_meas_0 - 0.06).abs() < f64::EPSILON);
        assert!((p_meas_1 - 0.07).abs() < f64::EPSILON);
        assert!((p1 - 0.08).abs() < f64::EPSILON);
        assert!((p2 - 0.09).abs() < f64::EPSILON);
    }

    #[test]
    fn test_uniform_probability() {
        // Test the uniform probability constructor
        let noise = BiasedDepolarizingNoiseModel::new_uniform(0.05);
        let (p_prep, p_meas_0, p_meas_1, p1, p2) = noise.probabilities();
        assert!((p_prep - 0.05).abs() < f64::EPSILON);
        assert!((p_meas_0 - 0.05).abs() < f64::EPSILON);
        assert!((p_meas_1 - 0.05).abs() < f64::EPSILON);
        assert!((p1 - 0.05).abs() < f64::EPSILON);
        assert!((p2 - 0.05).abs() < f64::EPSILON);

        // Test the uniform probability setter
        let mut noise = BiasedDepolarizingNoiseModel::new(0.01, 0.02, 0.03, 0.04, 0.05);
        noise.set_uniform_probability(0.07);
        let (p_prep, p_meas_0, p_meas_1, p1, p2) = noise.probabilities();
        assert!((p_prep - 0.07).abs() < f64::EPSILON);
        assert!((p_meas_0 - 0.07).abs() < f64::EPSILON);
        assert!((p_meas_1 - 0.07).abs() < f64::EPSILON);
        assert!((p1 - 0.07).abs() < f64::EPSILON);
        assert!((p2 - 0.07).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "Probability must be between 0.0 and 1.0")]
    fn test_invalid_probability_panics() {
        let mut noise = BiasedDepolarizingNoiseModel::new(0.1, 0.2, 0.3, 0.4, 0.5);
        noise.set_probabilities(0.1, 0.2, 1.1, 0.4, 0.5); // Should panic
    }

    #[test]
    fn test_builder() {
        // Create a noise model with the builder
        let noise = BiasedDepolarizingNoiseModel::builder()
            .with_prep_probability(0.1)
            .with_meas_0_probability(0.2)
            .with_meas_1_probability(0.3)
            .with_p1_probability(0.4)
            .with_p2_probability(0.5)
            .build();

        // Get the boxed noise model's probabilities using any_ref downcast
        let noise_ref = noise
            .as_any()
            .downcast_ref::<BiasedDepolarizingNoiseModel>()
            .unwrap();
        let (p_prep, p_meas_0, p_meas_1, p1, p2) = noise_ref.probabilities();

        assert!((p_prep - 0.1).abs() < f64::EPSILON);
        assert!((p_meas_0 - 0.2).abs() < f64::EPSILON);
        assert!((p_meas_1 - 0.3).abs() < f64::EPSILON);
        assert!((p1 - 0.4).abs() < f64::EPSILON);
        assert!((p2 - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_biased_measurement() {
        // Create a noise model with 100% flip probabilities for deterministic testing
        let mut noise = BiasedDepolarizingNoiseModel::new(0.0, 1.0, 1.0, 0.0, 0.0);

        // Test measurement bias - all 0s should be flipped to 1s
        assert_eq!(noise.apply_bias_to_measurement(0, 0), (0, 1));

        // Test measurement bias - all 1s should be flipped to 0s
        assert_eq!(noise.apply_bias_to_measurement(0, 1), (0, 0));

        // Create a noise model with 0% flip probabilities
        noise = BiasedDepolarizingNoiseModel::new(0.0, 0.0, 0.0, 0.0, 0.0);

        // Test measurement bias - all 0s should remain 0s
        assert_eq!(noise.apply_bias_to_measurement(0, 0), (0, 0));

        // Test measurement bias - all 1s should remain 1s
        assert_eq!(noise.apply_bias_to_measurement(0, 1), (0, 1));
    }
}
