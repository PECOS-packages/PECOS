// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

//! # General Noise Model Implementation
//!
//! This module implements a detailed noise model for quantum operations that simulates
//! realistic noise processes in quantum computing hardware, particularly ion trap systems.
//! The model is based on the Python implementation in `pecos.noise_models.general_noise`.
//!
//! ## Overview
//!
//! The `GeneralNoiseModel` provides:
//!
//! - Parameterized error rates for various quantum operations (preparation, measurement, gates)
//! - Support for leakage errors (qubits leaving the computational subspace)
//! - Emission errors that can cause leakage or Pauli-type noise
//! - Asymmetric measurement errors
//! - Angle-dependent error rates for certain gates (e.g., RZZ)
//! - Parameter scaling to convert between average and total error rates
//!
//! ## Physical Processes Modeled
//!
//! This noise model captures several important physical processes:
//!
//! - **Initialization errors**: Imperfect state preparation due to optical pumping errors
//! - **Measurement errors**: Asymmetric readout errors common in fluorescence detection
//! - **Gate errors**: Depolarizing and coherent errors during single and two-qubit operations
//! - **Leakage**: Transitions outside the computational basis (e.g., to higher energy levels)
//! - **Seepage**: Spontaneous return from leaked states to the computational basis
//! - **Emission errors**: Spontaneous emission events during gate operations
//!
//! ## Features from Python Implementation
//!
//! This Rust implementation includes most core features from the Python model:
//!
//! - Pauli error channels for single and two-qubit gates
//! - Leakage and emission error models
//! - Parameter scaling for error rates
//! - Angle-dependent noise for parameterized gates
//!
//! Some features from the Python implementation that are not yet fully implemented:
//!
//! - Crosstalk errors between nearby qubits
//! - Memory/idle noise with T1/T2 processes
//! - Repumping cycles for leaked qubits
//! - Zone-specific error rates
//! - Coherent vs. incoherent dephasing distinction
//!
//! ## Usage
//!
//! The noise model can be instantiated directly or through a builder pattern:
//!
//! ```rust
//! use pecos_engines::engines::noise::GeneralNoiseModel;
//!
//! // Using the builder with explicit error rates
//! let noise_model = GeneralNoiseModel::builder()
//!     .with_prep_probability(0.01)
//!     .with_meas_0_probability(0.02)
//!     .with_meas_1_probability(0.03)
//!     .with_single_qubit_probability(0.04)
//!     .with_two_qubit_probability(0.05)
//!     .with_seed(42)
//!     .build();
//! ```

#![allow(clippy::too_many_lines)]

use crate::byte_message::{ByteMessage, ByteMessageBuilder, QuantumGate, gate_type::GateType};
use crate::engines::noise::noise_rng::NoiseRng;
use crate::engines::noise::utils::NoiseUtils;
use crate::engines::noise::utils::ProbabilityValidator;
use crate::engines::noise::weighted_sampler::{
    SingleQubitWeightedSampler, TwoQubitWeightedSampler,
};
use crate::engines::noise::{NoiseModel, RngManageable};
use crate::engines::{ControlEngine, EngineStage};
use log::trace;
use pecos_core::errors::PecosError;
use rand_chacha::ChaCha8Rng;
use std::any::Any;
use std::collections::BTreeMap;
use std::collections::HashSet;

/// General noise model implementation that includes parameterized error channels for various quantum operations
///
/// This comprehensive noise model for quantum computers includes:
/// - **Initialization errors**: Errors during qubit preparation to |0⟩
/// - **Measurement errors**: Asymmetric bit flip errors during measurements
/// - **Gate errors**: Depolarizing and coherent errors during single and two-qubit operations
/// - **Memory errors**: Dephasing during idle periods
/// - **Leakage errors**: Transitions outside the computational subspace
/// - **Emission errors**: Non-unitary errors that can cause leakage
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct GeneralNoiseModel {
    /// Set of gate types that should not have noise applied
    ///
    /// Gates in this set may be those that are implemented in software rather than
    /// with physical operations, so no noise should be applied to them.
    noiseless_gates: HashSet<GateType>,

    /// Whether to replace leakage with depolarizing noise
    ///
    /// If true, instead of marking qubits as leaked, completely depolarizing noise will be applied.
    /// This is useful for studying the effects for comparing the effects of leakage vs.
    /// depolarizing noise.
    /// TODO: Consider making this more a float and becoming `leakage_scale`
    leak2depolar: bool,

    /// Probability of applying a fault during preparation (initialization)
    ///
    /// This parameter models faults that occur when initializing a qubit to |0⟩. In ion trap
    /// systems, this can correspond to imperfect optical pumping or faults in the initial
    /// state preparation process.
    p_prep: f64,

    /// Relative probability that a preparation fault leads to leakage
    ///
    /// Controls what fraction of preparation faults result in leakage out of the computational
    /// subspace. In ion trap systems, this could represent population in states other than the
    /// qubit states after initialization. Ranges from 0 to 1.
    p_prep_leak_ratio: f64,

    /// Probability of crosstalk during initialization operations
    ///
    /// Models the probability that an initialization operation on one qubit affects nearby qubits.
    /// In ion trap systems, this could represent scattered light during optical pumping affecting
    /// neighboring ions.
    p_prep_crosstalk: f64,

    /// Probability of applying a fault after single-qubit gates
    ///
    /// Models depolarizing channel + leakage noise for single-qubit gates.
    ///
    /// In physical systems, this represents coherent control errors, decoherence during gate
    /// operation, and other forms of noise affecting single-qubit operations.
    p1: f64,

    /// The proportion of single-qubit errors that are emission errors
    ///
    /// Controls what fraction of errors on single-qubit gates are emission errors (which can
    /// cause leakage) versus standard depolarizing errors. In ion trap systems, this could model
    /// spontaneous emission from excited states during gate operations. Ranges from 0 to 1.
    p1_emission_ratio: f64,

    /// Probability of a leaked qubit being seeped (released from leakage) for single-qubit gates if
    /// a spontaneous emission event occurs
    ///
    /// Models the rate at which qubits that have leaked from the computational subspace
    /// spontaneously return. In ion trap systems, this could represent decay from metastable
    /// states back to the computational subspace.
    p1_seepage_prob: f64,

    /// Probability model for Pauli faults on single qubit gates
    ///
    /// Specifies the distribution of different Pauli errors (X, Y, Z) that can occur.
    /// For a uniform depolarizing channel, each error type would have equal probability.
    ///
    /// The distribution is stored as pre-computed, cached sampler instead of the `HashMap` that is the input.
    p1_pauli_model: SingleQubitWeightedSampler,

    /// Probability model for emission errors on single qubit gates
    ///
    /// Specifies the distribution of different spontaneous emission error types that can occur.
    /// This includes errors that may cause state transitions outside the computational basis.
    ///
    /// The distribution is stored as pre-computed, cached sampler instead of the `HashMap` that is the input.
    p1_emission_model: SingleQubitWeightedSampler,

    /// Probability of applying a fault after two-qubit gates
    ///
    /// Models depolarizing channel + leakage noise for two-qubit gates.
    p2: f64,

    /// Scaling parameters for RZZ gate error rate - coefficient a
    ///
    /// Part of a parameterized model for angle-dependent errors in RZZ gates.
    /// The error rate is modeled as a function of angle θ: p(θ) = a + b|θ| + c|θ|^d
    przz_a: f64,

    /// Scaling parameters for RZZ gate error rate - coefficient b
    przz_b: f64,

    /// Scaling parameters for RZZ gate error rate - coefficient c
    przz_c: f64,

    /// Scaling parameters for RZZ gate error rate - coefficient d
    przz_d: f64,

    /// Power parameter for RZZ gate error rate scaling
    ///
    /// Controls how error probabilities scale with rotation angle in RZZ gates.
    /// Error scales as `theta^przz_power` where theta is the gate angle.
    /// Typically set to 1.0 for linear scaling.
    przz_power: f64,

    /// The proportion of two-qubit errors that are emission faults
    ///
    /// Controls what fraction of faults on two-qubit gates are spontaneous emission faults versus
    /// standard depolarizing faults. In ion trap systems, this could model decay or transitions to
    /// non-computational states during two-qubit operations. Ranges from 0 to 1.
    p2_emission_ratio: f64,

    /// Probability of a leaked qubit being seeped (released from leakage) for two-qubit gates if
    /// a spontaneous emission event occurs
    ///
    /// Models the rate at which qubits that have leaked from the computational subspace
    /// spontaneously return. In ion trap systems, this could represent decay from metastable
    /// states back to the computational subspace.
    p2_seepage_prob: f64,

    /// Probability model for Pauli errors on two-qubit gates
    ///
    /// Specifies the distribution of different two-qubit Pauli errors that can occur.
    /// For a uniform depolarizing channel, each of the 15 non-identity two-qubit Pauli
    /// operators would have equal probability.
    ///
    /// The distribution is stored as pre-computed, cached sampler instead of the `HashMap` that is the input.
    p2_pauli_model: TwoQubitWeightedSampler,

    /// Probability model for spontaneous emission errors on two-qubit gates
    ///
    /// Specifies the distribution of different emission error types that can occur during
    /// two-qubit operations. This includes errors that may cause state transitions outside
    /// the computational basis.
    ///
    /// The distribution is stored as pre-computed, cached sampler instead of the `HashMap` that is the input.
    p2_emission_model: TwoQubitWeightedSampler,

    /// Whether to use coherent dephasing vs incoherent (stochastic) dephasing
    ///
    /// If true, dephasing is modeled as coherent phase rotations using RZ gates.
    /// If false, dephasing is modeled as stochastic Z errors with quadratic scaling.
    ///
    /// In physical systems, coherent dephasing represents systematic phase evolution
    /// such as frequency offsets.
    coherent_dephasing: bool,

    /// Scaling factor to convert coherent dephasing rates to incoherent rates
    ///
    /// When using incoherent (stochastic) dephasing, this factor adjusts the dephasing rate. This
    /// is a fudge factor used to artificially increase the dephasing rate when modeling the
    /// quadratic dephasing stochastically since such modeling does not account for coherent
    /// effects.
    ///
    /// # Panics
    ///
    /// Panics if the factor is not positive (less than or equal to 0.0).
    coherent_to_incoherent_factor: f64,

    /// Whether to apply crosstalk on a per-gate basis
    ///
    /// If true, crosstalk is applied separately for each target qubit in a multi-qubit
    /// operation. If false, crosstalk is applied only once for the entire operation.
    /// TODO: consider separate per crosstalk channel
    crosstalk_per_gate: bool,

    /// Probability of flipping a 0 measurement to 1
    ///
    /// This asymmetric measurement error models cases when a qubit in state |0⟩ is incorrectly
    /// measured as 1.
    ///
    /// In ion trap systems, this may occur due to imperfect state detection or
    /// background counts during fluorescence detection.
    p_meas_0: f64,

    /// Probability of flipping a 1 measurement to 0
    ///
    /// This asymmetric measurement error models cases when a qubit in state |1⟩ is incorrectly
    /// measured as 0.
    ///
    /// In ion trap systems, this may occur due to decay during measurement or
    /// imperfect detection efficiency.
    p_meas_1: f64,

    /// Probability of crosstalk during measurement operations
    ///
    /// Models the probability that a measurement operation on one qubit affects nearby qubits. In
    /// ion trap systems, this could represent scattered light during fluorescence detection
    /// affecting neighboring ions.
    p_meas_crosstalk: f64,

    // --- internally used variables --- //
    /// The maximum of `p_meas_0` and `p_meas_1`
    ///
    /// Used to determine the overall measurement error rate.
    p_meas_max: f64,

    /// Set of qubits that are currently in a leaked state
    ///
    /// Tracks which qubits have leaked out of the computational subspace and are
    /// therefore not affected by computational gates but might still affect measurements.
    leaked_qubits: HashSet<usize>,

    /// Random number generator for stochastic noise processes
    rng: NoiseRng<ChaCha8Rng>,
}

impl ControlEngine for GeneralNoiseModel {
    type Input = ByteMessage;
    type Output = ByteMessage;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    /// Method first called at the start of the `QuantumSystem` processing a message
    fn start(
        &mut self,
        input: Self::Input,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        // Apply noise to the gates
        let noisy_gates = match self.apply_noise_on_start(&input) {
            Ok(gates) => gates,
            Err(e) => {
                return Err(PecosError::Processing(format!(
                    "Noise application error: {e}"
                )));
            }
        };

        // Return the noisy operations to QuantumEngine for processing/simulation
        Ok(EngineStage::NeedsProcessing(noisy_gates))
    }

    /// Method called when the `NoiseModel` has sent a message to its `QuantumEngine` and is
    /// receiving a message back. This gives an opportunity to react to the `QuantumEngine`.
    fn continue_processing(
        &mut self,
        msg: Self::EngineOutput,
    ) -> Result<EngineStage<Self::EngineInput, Self::Output>, PecosError> {
        // Apply biased measurement to measurement results
        trace!("GeneralNoise::continue_processing - applying biased measurement");
        let results = self
            .apply_noise_on_continue_processing(msg)
            .map_err(|e| PecosError::Processing(format!("Error processing noise: {e}")))?;

        // Calling Complete to signal that the NoiseModel is returning its msg back to the
        // QuantumSystem.
        Ok(EngineStage::Complete(results))
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        // Reset the noise model state
        self.reset_noise_model();
        Ok(())
    }
}

impl NoiseModel for GeneralNoiseModel {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl RngManageable for GeneralNoiseModel {
    type Rng = ChaCha8Rng;

    fn set_rng(&mut self, rng: Self::Rng) -> Result<(), PecosError> {
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

impl ProbabilityValidator for GeneralNoiseModel {}

impl GeneralNoiseModel {
    /// Create a new noise model with the specified error parameters
    ///
    /// Creates a `GeneralNoiseModel` with the specified error probabilities while using default values
    /// for all other parameters. This is a convenience method for cases where you only need to customize
    /// the basic error rates.
    ///
    /// * `p_prep` - Preparation (initialization) error probability
    /// * `p_meas_0` - Probability of measuring 1 when the state is |0⟩
    /// * `p_meas_1` - Probability of measuring 0 when the state is |1⟩
    /// * `p1` - Single-qubit gate error probability (average error rate)
    /// * `p2` - Two-qubit gate error probability (average error rate)
    ///
    /// For more extensive customization, use the builder pattern with `GeneralNoiseModel::builder()`.
    /// For default parameters, use `GeneralNoiseModel::default()`.
    ///
    /// # Example
    /// ```
    /// use pecos_engines::engines::noise::GeneralNoiseModel;
    ///
    /// // Create model with specified error probabilities
    /// let mut model = GeneralNoiseModel::new(0.01, 0.01, 0.01, 0.05, 0.1);
    /// ```
    #[must_use]
    pub fn new(p_prep: f64, p_meas_0: f64, p_meas_1: f64, p1: f64, p2: f64) -> Self {
        GeneralNoiseModel {
            p_prep,
            p1,
            p2,
            p_meas_0,
            p_meas_1,
            ..Default::default()
        }
    }

    /// Create a new builder for the general noise model
    #[must_use]
    pub fn builder() -> GeneralNoiseModelBuilder {
        GeneralNoiseModelBuilder::new()
    }

    /// Get the current error probabilities
    #[must_use]
    pub fn probabilities(&self) -> (f64, f64, f64, f64, f64, f64) {
        (
            self.p_prep,
            self.p_meas_0,
            self.p_meas_1,
            self.p1,
            self.p2,
            self.p_prep_leak_ratio,
        )
    }

    /// Apply noise at the start of `QuantumSystem` processing (typically a collection of gates)
    ///
    /// # Panics
    ///
    /// Panics if the input `ByteMessage` cannot be parsed as quantum operations.
    pub fn apply_noise_on_start(&mut self, input: &ByteMessage) -> Result<ByteMessage, String> {
        let mut builder = NoiseUtils::create_quantum_builder();
        let mut err = None;

        // Parse the input as quantum operations
        let gates = input
            .parse_quantum_operations()
            .expect("Failed to parse input as quantum operations");

        // TODO: Make this noise model handle gates that have multiple qubit arguments
        //       Currently it is assumed one gate per gate type...
        for gate in gates {
            // Skip noise application for noiseless gates
            if self.is_noiseless_gate(&gate.gate_type) {
                // Just add the gate as-is, without any noise
                // TODO: Still apply leakage rules
                builder.add_quantum_gate(&gate);
                trace!("Skipping noise for noiseless gate: {:?}", gate.gate_type);
                continue;
            }

            // For non-noiseless gates with qubits, we'll let the specific handlers
            // decide whether to add the original gate based on error models
            match gate.gate_type {
                GateType::Idle => {
                    // Still apply any noise that might result from idling
                    self.apply_idle_faults(&gate, &mut builder);
                    // Skip adding the Idle gate itself to the builder
                }
                GateType::Prep => {
                    self.apply_prep_faults(&gate, &mut builder);

                    // TODO: look closely at prep crosstalk...
                    // Potentially apply crosstalk
                    if self.p_prep_crosstalk > 0.0 {
                        self.prep_crosstalk(&gate.qubits, &mut builder);
                    }
                }
                GateType::R1XY
                | GateType::RZ
                | GateType::H
                | GateType::X
                | GateType::Y
                | GateType::Z
                | GateType::U => {
                    self.apply_sq_faults(&gate, &mut builder);
                }
                GateType::RZZ | GateType::SZZ | GateType::SZZdg | GateType::CX => {
                    // For RZZ gates, use angle-dependent error rates
                    let p2 = if gate.gate_type == GateType::RZZ {
                        let angle = gate.params[0];
                        self.rzz_error_rate(angle)
                    } else {
                        self.p2
                    };

                    self.apply_tq_faults(&gate, p2, &mut builder);
                }
                GateType::Measure => {
                    // Measurement noise is handled in apply_bias_to_message
                    // We still need to add the original gate here
                    builder.add_quantum_gate(&gate);
                }
                // This wildcard pattern is currently unreachable since all existing gate types
                // are handled in the cases above. We keep it as a safeguard for any future
                // gate types that might be added to the GateType enum.
                #[allow(unreachable_patterns)]
                _ => {
                    let err_msg = format!("Unsupported gate type: {:?}", gate.gate_type);
                    err = Some(err_msg);
                }
            }
        }

        if let Some(e) = err {
            return Err(e);
        }

        Ok(builder.build())
    }

    /// Apply measurement faults to the message after measurements have occurred
    ///
    /// This method applies several types of measurement noise:
    /// 1. Readout errors (asymmetric bit flips)
    /// 2. Handling of leaked qubits (ensuring they measure as 1)
    /// 3. Crosstalk effects on nearby qubits
    ///
    /// In physical systems, this represents detection errors, crosstalk, and special
    /// handling of qubit states outside the computational basis.
    pub fn apply_noise_on_continue_processing(
        &mut self,
        message: ByteMessage,
    ) -> Result<ByteMessage, PecosError> {
        // If there are no measurement results, return the message unchanged
        if !NoiseUtils::has_measurements(&message) {
            return Ok(message);
        }

        // Parse the measurements from the message
        let measurements = message.parse_measurements()?;
        if measurements.is_empty() {
            return Ok(message);
        }

        // extract qubit measurements
        let Ok(measurement_results) = message.measurement_results_as_vec() else {
            return Ok(ByteMessageBuilder::new().build());
        };

        // Get qubits that were measured
        let measured_qubits = message.parse_measured_qubits().unwrap_or_default();

        // Collect the measured qubits as usize for crosstalk
        let measured_qubits_usize: Vec<usize> =
            measured_qubits.iter().map(|&q| q as usize).collect();

        // Apply biases and handle leaked qubits
        let biased_results = self.apply_meas_faults(&measured_qubits_usize, &measurement_results);

        // TODO: Look closely at meas crosstalk...
        // Now check if we need to apply measurement crosstalk
        if !measured_qubits_usize.is_empty() && self.p_meas_crosstalk > 0.0 {
            // Create a new builder for quantum operations to hold crosstalk effects
            let mut operations_builder = ByteMessage::quantum_operations_builder();

            // Apply crosstalk to nearby qubits
            self.meas_crosstalk(&measured_qubits_usize, &mut operations_builder);

            // Build the operations message with crosstalk effects
            let operations_message = operations_builder.build();

            // If there are any operations from crosstalk, we need to return both messages
            if !operations_message.is_empty()? {
                trace!(
                    "Applied measurement crosstalk to qubits near {:?}",
                    measured_qubits_usize
                );

                // In a real integration, we would need to coordinate with the engine system
                // to ensure that both messages are processed correctly.
                // For now, we'll just return the measurement results since they're expected.
                // In a more comprehensive implementation, we'd need a way to queue both messages.
                return Ok(biased_results);
            }
        }

        // Return just the biased results if no crosstalk was applied
        Ok(biased_results)
    }

    /// Apply preparation (initialization) noise
    ///
    /// State prep noise model:
    /// 1. Reset all states including leaked qubits to |0⟩
    /// 2. With probability `p_prep` per qubit, the initialization fails
    /// 3. If failure occurs, with probability `p_prep_leak_ratio` the qubit leaks out of
    ///    computational space.
    /// 4. Otherwise, the qubit is prepared in the wrong state (|1⟩ instead of |0⟩)
    ///
    /// In ion trap systems, this models imperfect optical pumping or errors in the initial
    /// state preparation process that fails to correctly initialize the qubit.
    pub fn apply_prep_faults(&mut self, gate: &QuantumGate, builder: &mut ByteMessageBuilder) {
        // unleaking qubits - preparation resets leaked qubits to the zero state
        for &qubit in &gate.qubits {
            if self.is_leaked(qubit) {
                self.mark_as_unleaked(qubit);
                trace!("Qubit {} unleaked due to preparation", qubit);
            }
        }

        // Unlike SQ and TQ gates, state prep always occurs even if the qubit leaked
        builder.add_quantum_gate(gate);

        // Skip if probability is zero
        if self.p_prep <= 0.0 {
            return;
        }

        // Apply state prep faults
        for &qubit in &gate.qubits {
            // Apply bit-flip error with probability p_prep
            if self.rng.occurs(self.p_prep) {
                // Determine if this error should cause leakage
                if self.rng.occurs(self.p_prep_leak_ratio) {
                    if let Some(gate) = self.leak(qubit) {
                        builder.add_quantum_gate(&gate);
                    }
                    trace!("Qubit {} leaked during preparation", qubit);
                } else {
                    builder.add_x(&[qubit]);
                    trace!("Preparation error on qubit {}", qubit);
                }
            }
        }
    }

    /// Apply single-qubit gate noise faults
    ///
    /// Models errors that occur during single-qubit gate operations:
    /// 1. With probability p1, there is an error
    /// 2. If error occurs, with probability `p1_emission_ratio` it's a spontaneous emission error
    /// 3. Otherwise, it's a standard Pauli error (X, Y, Z)
    ///
    /// In physical systems, spontaneous emission errors can cause leakage out of the computational
    /// subspace, while Pauli errors represent standard decoherence and control errors.
    ///
    /// # Panics
    ///
    /// Panics if sampling from the Pauli model fails or if an invalid Pauli operator is encountered.
    pub fn apply_sq_faults(&mut self, gate: &QuantumGate, builder: &mut ByteMessageBuilder) {
        let mut noise = Vec::new();
        let mut removed_gates = false;
        let mut original_gate_qubits: Vec<usize> = Vec::new();

        for &qubit in &gate.qubits {
            // Track whether to add the original gate
            let mut add_original_gate = true;
            let has_leakage = self.is_leaked(qubit);

            if has_leakage {
                add_original_gate = false;
            }

            if self.rng.occurs(self.p1) {
                // Spontaneous emission
                if self.rng.occurs(self.p1_emission_ratio) {
                    // If qubit has leaked and spontaneous emission has occurred... seep the qubit
                    if has_leakage {
                        if let Some(gates) = self.seep(qubit, self.p1_seepage_prob) {
                            noise.extend(gates);
                        }
                    } else {
                        add_original_gate = false;

                        let result = self.p1_emission_model.sample_gates(&mut self.rng, qubit);

                        if result.has_leakage() {
                            // Handle leakage
                            if let Some(gate) = self.leak(qubit) {
                                noise.push(gate);
                            }
                        } else if let Some(gate) = result.gate {
                            // Handle Pauli gate
                            noise.push(gate);
                            trace!("Applied Pauli error to qubit {}", qubit);
                        }
                    }
                } else if !has_leakage {
                    // Pauli noise
                    let result = self.p1_pauli_model.sample_gates(&mut self.rng, qubit);
                    if let Some(gate) = result.gate {
                        noise.push(gate);
                        trace!("Applied Pauli error to qubit {}", qubit);
                    }
                }
            }

            // Add the original gate only if there were no leakage errors
            if add_original_gate {
                original_gate_qubits.push(qubit);
            } else {
                removed_gates = true;
            }
        }

        if removed_gates {
            // There are some gates left to add
            if !original_gate_qubits.is_empty() {
                let new_gate = QuantumGate::new(
                    gate.gate_type,
                    original_gate_qubits,
                    gate.params.clone(),
                    None,
                );
                builder.add_quantum_gate(&new_gate);
            }
        } else {
            builder.add_quantum_gate(gate);
        }

        if !noise.is_empty() {
            builder.add_quantum_gates(&noise);
        }
    }

    /// Apply two-qubit gate noise faults
    ///
    /// Models errors that occur during two-qubit gate operations:
    /// 1. With probability p2, there is an error
    /// 2. If error occurs, with probability `p2_emission_ratio` it's an spontaneous emission error
    /// 3. Otherwise, it's a standard two-qubit Pauli error (IX, IY, IZ, XI, ...)
    ///
    /// In physical systems, emission errors can cause leakage, while Pauli errors
    /// represent standard decoherence, cross-talk, and control errors.
    ///
    /// # Panics
    ///
    /// Panics if sampling from the Pauli model fails or if an invalid Pauli operator is encountered.
    pub fn apply_tq_faults(
        &mut self,
        gate: &QuantumGate,
        p: f64,
        builder: &mut ByteMessageBuilder,
    ) {
        let mut noise = Vec::new();
        let mut removed_gates = false;
        let mut original_gate_qubits: Vec<usize> = Vec::new();

        for qubits in gate.qubits.chunks_exact(2) {
            let mut add_original_gate = true;

            // Check if the gate is acting on a leaked qubit in a way to
            let has_leakage = !self.leaked_qubits.is_empty()
                && gate.qubits.iter().any(|&qubit| self.is_leaked(qubit));

            if has_leakage {
                add_original_gate = false;
            }

            if self.rng.occurs(p) {
                if self.rng.occurs(self.p2_emission_ratio) {
                    if has_leakage {
                        // potentially seep qubits
                        for qubit in &gate.qubits {
                            if self.is_leaked(*qubit) {
                                if let Some(gates) = self.seep(*qubit, self.p2_seepage_prob) {
                                    noise.extend(gates);
                                }
                            }
                        }
                    } else {
                        // Spontaneous emission noise
                        add_original_gate = false;

                        let result = self.p2_emission_model.sample_gates(
                            &mut self.rng,
                            qubits[0],
                            qubits[1],
                        );

                        if result.has_leakage() {
                            for (qubit, leaked) in qubits.iter().zip(result.has_leakages().iter()) {
                                if *leaked {
                                    if let Some(gate) = self.leak(*qubit) {
                                        noise.push(gate);
                                    }
                                }
                            }
                        }

                        if let Some(gates) = result.gates {
                            noise.extend(gates);
                            trace!(
                                "Applied Pauli error to qubits {} and {}",
                                qubits[0], qubits[1]
                            );
                        }
                    }
                } else if !has_leakage {
                    // Pauli noise
                    let result =
                        self.p2_pauli_model
                            .sample_gates(&mut self.rng, qubits[0], qubits[1]);
                    if let Some(gates) = result.gates {
                        noise.extend(gates);
                        trace!(
                            "Applied Pauli error to qubits {} and {}",
                            qubits[0], qubits[1]
                        );
                    }
                }
            }

            if add_original_gate {
                original_gate_qubits.extend(qubits);
            } else {
                removed_gates = true;
            }
        }

        if removed_gates {
            // There are some gates left to add
            if !original_gate_qubits.is_empty() {
                let new_gate = QuantumGate::new(
                    gate.gate_type,
                    original_gate_qubits,
                    gate.params.clone(),
                    None,
                );
                builder.add_quantum_gate(&new_gate);
            }
        } else {
            builder.add_quantum_gate(gate);
        }

        builder.add_quantum_gates(&noise);
    }

    /// Apply measurement bias and handle leaked qubits
    ///
    /// This method handles two specific types of measurement faults:
    /// 1. Asymmetric readout errors based on `p_meas_0` and `p_meas_1`
    /// 2. Special handling for leaked qubits (ensuring they measure as 1 + measurement noise)
    ///
    /// Returns a `ByteMessage` containing the biased measurement results
    pub fn apply_meas_faults(
        &mut self,
        measured_qubits: &[usize],
        measurement_results: &[(usize, u32)],
    ) -> ByteMessage {
        // TODO: Consider factoring out an overall measurement error rate p_meas and relative bias rates
        let mut results_builder = ByteMessage::measurement_results_builder();

        // Check if there are any leaked qubits
        let has_leakage = !self.leaked_qubits.is_empty()
            && measured_qubits.iter().any(|&qubit| self.is_leaked(qubit));

        for (&qubit, &(result_id, result)) in measured_qubits.iter().zip(measurement_results.iter())
        {
            let mut val = result;
            if has_leakage && self.is_leaked(qubit) {
                trace!("Unleaking qubit {} after measurement", qubit);
                self.mark_as_unleaked(qubit);
                // Force the measurement outcome to be 1 for previously leaked qubits
                val = 1;
                // But still apply biased measurement noise
                if self.rng.occurs(self.p_meas_1) {
                    trace!(
                        "Flipped measurement outcome of leakage 1->0 for result_id {}",
                        result_id
                    );
                    val = 0;
                }
            } else {
                // Potentially flip the measurement results
                if val == 1 {
                    if self.rng.occurs(self.p_meas_1) {
                        trace!(
                            "Flipped measurement outcome 0->1 for result_id {}",
                            result_id
                        );
                        val = 0;
                    }
                } else {
                    trace!(
                        "Flipped measurement outcome 1->0 for result_id {}",
                        result_id
                    );
                    if self.rng.occurs(self.p_meas_0) {
                        val = 1;
                    }
                }
            }
            results_builder.add_measurement_results(&[val as usize], &[result_id]);
        }

        // TODO: If qubits are in |1>, leak them again with some probability.
        //       Maybe move L -> |1> + noise to first round of noise...

        // Get the biased measurement results
        results_builder.build()
    }

    /// Apply idle qubit noise faults
    ///
    /// Models errors that occur during idle periods when qubits are not actively being manipulated:
    /// 1. Coherent dephasing: Phase rotation errors that accumulate during idle time
    /// 2. Incoherent dephasing: Stochastic Z errors
    ///
    /// The error rates scale with the idle duration, and are affected by `memory_scale` parameter.
    /// In physical systems, this sensitivity to the surrounding magnetic fields, represents
    /// heating, T2 decoherence, and other environmental interactions that affect the qubit while
    /// it's not being actively controlled.
    #[allow(clippy::unused_self)]
    pub fn apply_idle_faults(&mut self, _gate: &QuantumGate, _builder: &mut ByteMessageBuilder) {
        // let duration = gate.idle_duration();
        //
        // // Skip if duration is too small
        // if duration < f64::EPSILON {
        //     // Just pass through the gate without noise
        //     builder.add_quantum_gate(gate);
        //     return;
        // }
        //
        // // Filter out leaked qubits
        // let qubits: Vec<usize> = gate
        //     .qubits
        //     .iter()
        //     .filter(|&&q| !self.is_leaked(q))
        //     .copied()
        //     .collect();
        //
        // if qubits.is_empty() {
        //     return;
        // }
        //
        // // Call the existing dephasing method to apply the appropriate noise
        // // This will use the same dephasing model as other memory operations
        // self.apply_dephasing(
        //     builder,
        //     gate,
        //     duration,
        //     // For coherent dephasing
        //     Some(dephasing_rate),
        //     // For incoherent dephasing
        //     Some(dephasing_rate),
        //     // Whether to use coherent dephasing
        //     self.coherent_dephasing,
        // );
    }

    /// Mark a qubit as leaked
    ///
    /// When a qubit leaks, it moves outside the computational subspace and can no longer be
    /// affected by quantum gates, but may still be re-prepared and measured
    fn leak(&mut self, qubit: usize) -> Option<QuantumGate> {
        if self.leak2depolar {
            // Apply completely depolarizing noise instead of leakage
            trace!("Replaced leakage with Pauli error on qubit {}", qubit);
            self.rng.random_pauli_or_none(qubit)
        } else {
            // Mark qubit as leaked
            trace!("Marking qubit {} as leaked", qubit);
            self.mark_as_leaked(qubit);
            Some(QuantumGate::prep(qubit))
        }
    }

    fn mark_as_leaked(&mut self, qubit: usize) {
        // TODO: see if some of the mark_as_leaked needs to move to self.leak()
        trace!("Marking qubit {} as leaked", qubit);
        self.leaked_qubits.insert(qubit);
    }

    /// Check if a qubit is in a leaked state
    ///
    /// Returns true if the qubit has leaked from the computational subspace.
    fn is_leaked(&self, qubit: usize) -> bool {
        self.leaked_qubits.contains(&qubit)
    }

    /// Mark a qubit as no longer leaked, returning it to the computational subspace
    fn mark_as_unleaked(&mut self, qubit: usize) {
        self.leaked_qubits.remove(&qubit);
    }

    fn unleak(&mut self, qubit: usize) -> Option<QuantumGate> {
        trace!("Replaced leakage with Pauli error on qubit {}", qubit);
        if self.leak2depolar {
            None
        } else {
            trace!("Marking qubit {} as unleaked", qubit);
            self.mark_as_unleaked(qubit);
            Option::from(QuantumGate::prep(qubit))
        }
    }

    fn unleak_random_bit(&mut self, qubit: usize) -> Vec<QuantumGate> {
        let mut noise = vec![];

        if let Some(gate) = self.unleak(qubit) {
            noise.push(gate);
        }

        if let Some(gate) = self.rng.random_pauli_or_none(qubit) {
            noise.push(gate);
        }

        noise
    }

    fn seep(&mut self, qubit: usize, seepage_prob: f64) -> Option<Vec<QuantumGate>> {
        if self.rng.occurs(seepage_prob) {
            Option::from(self.unleak_random_bit(qubit))
        } else {
            None
        }
    }

    /// Reset the noise model for a new shot
    fn reset_noise_model(&mut self) {
        // Clear leaked qubits
        self.leaked_qubits.clear();
        // RNG state is intentionally not reset to maintain natural randomness
    }

    /// Apply coherent dephasing noise to a gate
    ///
    /// This method implements coherent phase rotation (systematic Z-rotation) noise
    /// that occurs during idle periods or during gates with a specified duration.
    ///
    /// In physical systems, coherent dephasing represents:
    /// - Systematic phase errors due to energy level shifts
    /// - Frequency offsets in control fields
    /// - AC Stark shifts
    /// - Other systematic Z-rotation errors
    ///
    /// # Parameters
    /// * `builder` - The `ByteMessageBuilder` to add gate operations to
    /// * `gate` - The gate experiencing dephasing
    /// * `duration` - The time duration over which dephasing occurs
    /// * `rate` - The dephasing rate parameter
    #[allow(dead_code)]
    fn apply_coherent_dephasing(
        &mut self,
        builder: &mut ByteMessageBuilder,
        gate: &QuantumGate,
        duration: f64,
        rate: f64,
    ) {
        // Only apply to qubits that are not in a leaked state
        let qubits: Vec<usize> = gate
            .qubits
            .iter()
            .filter(|&&q| !self.is_leaked(q))
            .copied()
            .collect();

        // If there are qubits to apply dephasing to, add a rotation gate
        if !qubits.is_empty() {
            // Create an RZ gate with angle determined by rate * duration
            let dephase_gate =
                QuantumGate::new(GateType::RZ, qubits.clone(), vec![rate * duration], None);

            // Add the gate to the circuit
            NoiseUtils::add_gate_to_builder(builder, &dephase_gate);

            trace!(
                "Applied coherent dephasing to qubits {:?} with angle {}",
                dephase_gate.qubits,
                rate * duration
            );
        }
    }

    /// Apply incoherent dephasing noise to a gate
    ///
    /// This method implements stochastic phase flip (Z) noise that occurs during
    /// idle periods or during gates with a specified duration. The noise can be
    /// scaled either linearly or quadratically with time.
    ///
    /// In physical systems, incoherent dephasing represents:
    /// - Random phase kicks from the environment
    /// - T2 decoherence processes
    /// - Fast magnetic field fluctuations
    /// - Thermal noise affecting energy levels
    ///
    /// # Parameters
    /// * `builder` - The `ByteMessageBuilder` to add gate operations to
    /// * `gate` - The gate experiencing dephasing
    /// * `duration` - The time duration over which dephasing occurs
    /// * `rate` - The dephasing rate parameter
    /// * `linear` - If true, scale linearly with time; if false, scale quadratically
    #[allow(dead_code)]
    fn apply_incoherent_dephasing(
        &mut self,
        builder: &mut ByteMessageBuilder,
        gate: &QuantumGate,
        duration: f64,
        rate: f64,
        linear: bool,
    ) {
        // Calculate dephasing probability
        let mut p_deph = rate * duration;

        // Apply quadratic scaling if not linear
        if !linear {
            p_deph = (p_deph.sin()).powi(2);
        }

        // Only proceed if there's a non-zero dephasing probability
        if p_deph > 0.0 {
            // Get non-leaked qubits
            let qubits: Vec<usize> = gate
                .qubits
                .iter()
                .filter(|&&q| !self.is_leaked(q))
                .copied()
                .collect();

            // Apply Z errors with probability p_deph
            for &qubit in &qubits {
                if self.rng.occurs(p_deph) {
                    // Apply a Z gate to represent a phase flip
                    let z_gate = QuantumGate::new(GateType::Z, vec![qubit], vec![], None);

                    NoiseUtils::add_gate_to_builder(builder, &z_gate);
                    trace!("Applied incoherent dephasing (Z error) to qubit {}", qubit);
                }
            }
        }
    }

    /// Apply general dephasing noise to a gate
    ///
    /// This is the main entry point for applying dephasing noise. It delegates to either
    /// coherent or incoherent dephasing methods based on the noise model parameters.
    /// It can also apply both types if needed.
    ///
    /// # Parameters
    /// * `builder` - The `ByteMessageBuilder` to add gate operations to
    /// * `gate` - The gate experiencing dephasing
    /// * `duration` - The time duration over which dephasing occurs
    /// * `coherent_rate` - Rate parameter for coherent dephasing (if applicable)
    /// * `incoherent_rate` - Rate parameter for incoherent dephasing (if applicable)
    /// * `use_coherent` - Whether to use coherent dephasing, overrides model's setting
    #[allow(dead_code)]
    fn apply_dephasing(
        &mut self,
        builder: &mut ByteMessageBuilder,
        gate: &QuantumGate,
        duration: f64,
        coherent_rate: Option<f64>,
        incoherent_rate: Option<f64>,
        use_coherent: bool,
    ) {
        // Apply coherent dephasing if enabled and rate is provided
        if use_coherent {
            if let Some(rate) = coherent_rate {
                // Use RZ gates for coherent dephasing
                for &qubit in &gate.qubits {
                    if !self.is_leaked(qubit) {
                        // Create RZ rotation with angle = rate * duration
                        builder.add_rz(rate * duration, &[qubit]);
                        trace!(
                            "Applied coherent dephasing to qubit {} with angle {}",
                            qubit,
                            rate * duration
                        );
                    }
                }
            }
        } else {
            // Apply quadratic incoherent dephasing
            if let Some(rate) = coherent_rate {
                // When using incoherent dephasing, apply the conversion factor
                let adjusted_rate = rate * self.coherent_to_incoherent_factor;
                let mut p_deph = adjusted_rate * duration;

                // Apply quadratic scaling
                p_deph = (p_deph.sin()).powi(2);

                // Apply Z errors with probability p_deph
                for &qubit in &gate.qubits {
                    if !self.is_leaked(qubit) && self.rng.occurs(p_deph) {
                        // Apply Z gate for phase error
                        builder.add_z(&[qubit]);
                        trace!(
                            "Applied incoherent dephasing (Z error) to qubit {} with probability {}",
                            qubit, p_deph
                        );
                    }
                }
            }
        }

        // Apply additional linear incoherent dephasing if rate is provided
        if let Some(rate) = incoherent_rate {
            let p_deph = rate * duration; // Linear scaling

            // Apply Z errors with probability p_deph
            for &qubit in &gate.qubits {
                if !self.is_leaked(qubit) && self.rng.occurs(p_deph) {
                    // Apply Z gate for phase error
                    builder.add_z(&[qubit]);
                    trace!(
                        "Applied linear incoherent dephasing (Z error) to qubit {}",
                        qubit
                    );
                }
            }
        }
    }

    /// Create a new method to handle requesting nearby qubits for crosstalk
    #[allow(dead_code)]
    fn get_nearby_qubits_for_crosstalk(_source_qubits: &[usize], _num_qubits: usize) -> Vec<usize> {
        // PLACEHOLDER: This will eventually request information from the ClassicalEngine
        // via the EngineSystem to get the nearest qubits based on device topology
        todo!()
    }

    // Replace the meas_crosstalk method to use the correct API
    #[allow(clippy::unused_self)]
    fn meas_crosstalk(&mut self, _locations: &[usize], _builder: &mut ByteMessageBuilder) {
        // placeholder
    }

    // Replace the prep_crosstalk method to use the correct API
    #[allow(clippy::unused_self)]
    fn prep_crosstalk(&mut self, _locations: &[usize], _builder: &mut ByteMessageBuilder) {
        // placeholder
    }

    /// Calculate the RZZ gate error rate based on the rotation angle
    ///
    /// with additional support for asymmetric scaling and power-law scaling
    /// Includes scaling by p2 (two-qubit gate error probability) to match Python implementation
    #[must_use]
    pub fn rzz_error_rate(&self, angle: f64) -> f64 {
        // Normalize angle by π - convert to a value in [0, 1] range
        let theta = angle.abs() / std::f64::consts::PI;

        // Apply power scaling to the normalized theta
        let theta_power = theta.powf(self.przz_power);

        // Determine base rate based on angle sign
        let base_rate = if angle < 0.0 {
            // Negative angle - use a and b coefficients
            self.przz_a * theta_power + self.przz_b
        } else if angle > 0.0 {
            // Positive angle - use c and d coefficients
            self.przz_c * theta_power + self.przz_d
        } else {
            // Angle is exactly zero - use average of b and d
            (self.przz_b + self.przz_d) * 0.5
        };

        base_rate * self.p2
    }

    /// Add a gate type to the set of noiseless gates
    ///
    /// Gates in this set will not have noise applied to them.
    ///
    /// # Parameters
    /// * `gate_type` - The type of gate to add to the noiseless gates set
    pub fn add_noiseless_gate(&mut self, gate_type: GateType) {
        self.noiseless_gates.insert(gate_type);
    }

    /// Remove a gate type from the set of noiseless gates
    ///
    /// # Parameters
    /// * `gate_type` - The type of gate to remove from the noiseless gates set
    pub fn remove_noiseless_gate(&mut self, gate_type: GateType) {
        self.noiseless_gates.remove(&gate_type);
    }

    /// Clear the set of noiseless gates
    pub fn clear_noiseless_gates(&mut self) {
        self.noiseless_gates.clear();
    }

    /// Check if a gate type is in the set of noiseless gates
    ///
    /// # Parameters
    /// * `gate_type` - The type of gate to check
    ///
    /// # Returns
    /// `true` if the gate is in the noiseless gates set, `false` otherwise
    #[must_use]
    pub fn is_noiseless_gate(&self, gate_type: &GateType) -> bool {
        self.noiseless_gates.contains(gate_type)
    }

    /// Accessor for the p1 Pauli distribution
    #[must_use]
    pub fn p1_pauli_model(&self) -> &SingleQubitWeightedSampler {
        &self.p1_pauli_model
    }

    /// Accessor for the p1 emission model
    #[must_use]
    pub fn p1_emission_model(&self) -> &SingleQubitWeightedSampler {
        &self.p1_emission_model
    }

    /// Accessor for the p2 Pauli model
    #[must_use]
    pub fn p2_pauli_model(&self) -> &TwoQubitWeightedSampler {
        &self.p2_pauli_model
    }

    /// Accessor for the p2 emission model
    #[must_use]
    pub fn p2_emission_model(&self) -> &TwoQubitWeightedSampler {
        &self.p2_emission_model
    }

    /// Reset the noise model and then set a new seed for the RNG
    ///
    /// This method rebuilds the noise model with the same parameters but a new seed,
    /// using the builder pattern.
    ///
    /// # Parameters
    /// * `seed` - The seed to set for the RNG
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn reset_with_seed(&mut self, seed: u64) -> Result<(), PecosError> {
        // First reset the noise model
        self.reset_noise_model();
        // Then set the seed
        self.set_seed(seed)
    }
}

/// Builder for creating general noise models
pub struct GeneralNoiseModelBuilder {
    p_prep: Option<f64>,
    p_meas_0: Option<f64>,
    p_meas_1: Option<f64>,
    p1: Option<f64>,
    p2: Option<f64>,
    p1_emission_ratio: Option<f64>,
    p2_emission_ratio: Option<f64>,
    p1_pauli_model: Option<SingleQubitWeightedSampler>,
    p1_emission_model: Option<SingleQubitWeightedSampler>,
    p2_pauli_model: Option<TwoQubitWeightedSampler>,
    p2_emission_model: Option<TwoQubitWeightedSampler>,
    p_prep_leak_ratio: Option<f64>,
    p1_seepage_prob: Option<f64>,
    p2_seepage_prob: Option<f64>,
    seed: Option<u64>,
    scale: Option<f64>,
    memory_scale: Option<f64>,
    prep_scale: Option<f64>,
    meas_scale: Option<f64>,
    leakage_scale: Option<f64>,
    p1_scale: Option<f64>,
    p2_scale: Option<f64>,
    emission_scale: Option<f64>,
    p_meas_crosstalk: Option<f64>,
    p_prep_crosstalk: Option<f64>,
    p_meas_crosstalk_scale: Option<f64>,
    p_prep_crosstalk_scale: Option<f64>,
    crosstalk_per_gate: Option<bool>,
    coherent_dephasing: Option<bool>,
    coherent_to_incoherent_factor: Option<f64>,
    przz_params: Option<(f64, f64, f64, f64)>,
    przz_power: Option<f64>,
    noiseless_gates: Option<HashSet<GateType>>,
    leak2depolar: Option<bool>,
}

impl Default for GeneralNoiseModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneralNoiseModelBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            p_prep: None,
            p_meas_0: None,
            p_meas_1: None,
            p1: None,
            p2: None,
            p1_emission_ratio: None,
            p2_emission_ratio: None,
            p1_pauli_model: None,
            p1_emission_model: None,
            p2_pauli_model: None,
            p2_emission_model: None,
            p_prep_leak_ratio: None,
            p1_seepage_prob: None,
            p2_seepage_prob: None,
            seed: None,
            scale: None,
            memory_scale: None,
            prep_scale: None,
            meas_scale: None,
            leakage_scale: None,
            p1_scale: None,
            p2_scale: None,
            emission_scale: None,
            p_meas_crosstalk: None,
            p_prep_crosstalk: None,
            p_meas_crosstalk_scale: None,
            p_prep_crosstalk_scale: None,
            crosstalk_per_gate: None,
            coherent_dephasing: None,
            coherent_to_incoherent_factor: None,
            przz_params: None,
            przz_power: None,
            noiseless_gates: None,
            leak2depolar: None,
        }
    }

    /// Validate that a value is a valid probability (between 0 and 1)
    fn validate_probability(prob: f64) -> f64 {
        assert!(
            (0.0..=1.0).contains(&prob),
            "Probability must be between 0 and 1, got {prob}"
        );
        prob
    }

    /// Validate that a value is positive
    fn validate_positive(value: f64, name: &str) -> f64 {
        assert!(value > 0.0, "{name} must be positive, got {value}");
        value
    }

    /// Validate that a value is non-negative
    fn validate_non_negative(value: f64, name: &str) -> f64 {
        assert!(value >= 0.0, "{name} must be non-negative, got {value}");
        value
    }

    /// Set the probability of error during preparation
    #[must_use]
    pub fn with_prep_probability(mut self, probability: f64) -> Self {
        self.p_prep = Some(Self::validate_probability(probability));
        self
    }

    /// Set the probability of bit flipping the measurement result
    #[must_use]
    pub fn with_meas_probability(mut self, probability: f64) -> Self {
        self.p_meas_0 = Some(Self::validate_probability(probability));
        self.p_meas_1 = Some(Self::validate_probability(probability));
        self
    }

    /// Set the probability of flipping 0 to 1 during measurement
    #[must_use]
    pub fn with_meas_0_probability(mut self, probability: f64) -> Self {
        self.p_meas_0 = Some(Self::validate_probability(probability));
        self
    }

    /// Set the probability of flipping 1 to 0 during measurement
    #[must_use]
    pub fn with_meas_1_probability(mut self, probability: f64) -> Self {
        self.p_meas_1 = Some(Self::validate_probability(probability));
        self
    }

    /// Set the average probability of error after single-qubit gates
    ///
    /// Rescaling from average error to total error
    ///
    /// This conversion is necessary because experiments report average error rates,
    /// but our noise models use total error rates.
    ///
    /// For a single-qubit gate with uniform error distribution across 3 Pauli errors,
    /// the ratio of total error rate to average error rate is 3/2.
    #[must_use]
    pub fn with_average_p1_probability(mut self, probability: f64) -> Self {
        self.p1 = Some(Self::validate_probability(probability * 3.0 / 2.0));
        self
    }

    /// Set the probability of error after single-qubit gates
    #[must_use]
    pub fn with_p1_probability(mut self, probability: f64) -> Self {
        self.p1 = Some(Self::validate_probability(probability));
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
    ///
    /// Rescaling from average error to total error
    ///
    /// This conversion is necessary because experiments report average error rates,
    /// but our noise models use total error rates.
    ///
    /// For a two-qubit gate with uniform error distribution across 15 Pauli errors,
    /// the ratio of total error rate to average error rate is 5/4.
    #[must_use]
    pub fn with_average_p2_probability(mut self, probability: f64) -> Self {
        self.p2 = Some(Self::validate_probability(probability * 5.0 / 4.0));
        self
    }

    /// Set the probability of error after two-qubit gates
    #[must_use]
    pub fn with_p2_probability(mut self, probability: f64) -> Self {
        self.p2 = Some(Self::validate_probability(probability));
        self
    }

    /// Set the probability of error after two-qubit gates
    ///
    /// This is an alias for `with_p2_probability` for API consistency.
    #[must_use]
    pub fn with_two_qubit_probability(self, probability: f64) -> Self {
        self.with_p2_probability(probability)
    }

    /// Set the Pauli error model for single-qubit gates
    #[must_use]
    pub fn with_p1_pauli_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        self.p1_pauli_model = Some(SingleQubitWeightedSampler::new(model));
        self
    }

    /// Set the emission error model for single-qubit gates
    #[must_use]
    pub fn with_p1_emission_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        self.p1_emission_model = Some(SingleQubitWeightedSampler::new(model));
        self
    }

    /// Set the preparation leakage ratio
    #[must_use]
    pub fn with_prep_leak_ratio(mut self, ratio: f64) -> Self {
        self.p_prep_leak_ratio = Some(Self::validate_probability(ratio));
        self
    }

    /// Set the seed for the random number generator
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the overall scaling factor for error probabilities
    ///
    /// A global multiplier applied to all error rates. This allows easy adjustment of the
    /// overall noise level without changing individual parameters. Typically used to
    /// simulate different device qualities or to study the effect of noise strength.
    #[must_use]
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
        self
    }

    /// Set the scaling factor for memory errors
    ///
    /// Controls the strength of errors that occur during idle periods or memory operations.
    /// In ion trap systems, this could represent heating or dephasing during storage times.
    #[must_use]
    pub fn with_memory_scale(mut self, scale: f64) -> Self {
        self.memory_scale = Some(scale);
        self
    }

    /// Set the scaling factor for initialization errors
    ///
    /// Multiplier for preparation error probabilities. Allows adjustment of the relative
    /// strength of initialization errors compared to other error types.
    #[must_use]
    pub fn with_prep_scale(mut self, scale: f64) -> Self {
        self.prep_scale = Some(scale);
        self
    }

    /// Set the scaling factor for measurement faults
    ///
    /// Multiplier for measurement error probabilities. Allows adjustment of the relative
    /// strength of readout errors compared to other error types.
    #[must_use]
    pub fn with_meas_scale(mut self, scale: f64) -> Self {
        self.meas_scale = Some(scale);
        self
    }

    /// Set the scaling factor for leakage errors
    ///
    /// Multiplier for leakage-related error probabilities. Controls how likely qubits
    /// are to transition outside the computational subspace during various operations.
    #[must_use]
    pub fn with_leakage_scale(mut self, scale: f64) -> Self {
        self.leakage_scale = Some(scale);
        self
    }

    /// Set the scaling factor for single-qubit gate errors
    ///
    /// Multiplier for single-qubit gate error probabilities. Allows adjustment of the
    /// relative strength of single-qubit gate errors compared to other error types.
    #[must_use]
    pub fn with_p1_scale(mut self, scale: f64) -> Self {
        self.p1_scale = Some(scale);
        self
    }

    /// Set the scaling factor for two-qubit gate errors
    ///
    /// Multiplier for two-qubit gate error probabilities. Allows adjustment of the relative
    /// strength of two-qubit gate errors compared to other error types. In most quantum
    /// technologies, two-qubit gates are typically more error-prone than single-qubit gates.
    #[must_use]
    pub fn with_p2_scale(mut self, scale: f64) -> Self {
        self.p2_scale = Some(scale);
        self
    }

    /// Set the scaling factor for spontaneous emission errors
    ///
    /// Multiplier for spontaneous-emission-related error probabilities. Controls the relative
    /// strength of errors that involve transitions outside the standard computational basis.
    /// TODO: consider replacing with leak2depolar
    #[must_use]
    pub fn with_emission_scale(mut self, scale: f64) -> Self {
        self.emission_scale = Some(scale);
        self
    }

    /// Set whether to use coherent dephasing
    #[must_use]
    pub fn with_coherent_dephasing(mut self, use_coherent: bool) -> Self {
        self.coherent_dephasing = Some(use_coherent);
        self
    }

    /// Set the coherent-to-incoherent conversion factor
    ///
    /// # Parameters
    /// * `factor` - The conversion factor between coherent and incoherent dephasing rates
    #[must_use]
    pub fn with_coherent_to_incoherent_factor(mut self, factor: f64) -> Self {
        self.coherent_to_incoherent_factor = Some(Self::validate_positive(
            factor,
            "Coherent-to-incoherent factor",
        ));
        self
    }

    /// Set RZZ parameter scaling for angle dependent error.
    ///
    /// The PECOS gate set has a parameterized-angle ZZ gate, RZZ(θ). For implementation
    /// Certain parameters relate to the strength of the asymmetric
    /// depolarizing noise. These parameters depend on the angle θ and are normalized so that
    /// θ = π/2 gives the 2-qubit fault probability (p2).
    ///
    /// The parameters for asymmetric depolarizing noise are fit parameters that model how the
    /// noise changes as the angle θ changes according to these equations:
    ///
    /// For θ < 0:
    ///     (`przz_a` × (|`θ|/π)^przz_power` + `przz_b`) × p2
    ///
    /// For θ > 0:
    ///     (`przz_c` × (|`θ|/π)^przz_power` + `przz_d`) × p2
    ///
    /// For θ = 0:
    ///     (`przz_b` + `przz_d`) × 0.5 × p2
    ///
    /// # Parameters
    /// * `a` - Coefficient for scaling negative angles (`przz_a`)
    /// * `b` - Offset for negative angles (`przz_b`)
    /// * `c` - Coefficient for scaling positive angles (`przz_c`)
    /// * `d` - Offset for positive angles (`przz_d`)
    #[must_use]
    pub fn with_przz_params(mut self, a: f64, b: f64, c: f64, d: f64) -> Self {
        self.przz_params = Some((a, b, c, d));
        self
    }

    /// Set power parameter for RZZ error scaling
    ///
    /// # Parameters
    /// * `power` - The power to which theta is raised in the RZZ error rate formula
    #[must_use]
    pub fn with_przz_power(mut self, power: f64) -> Self {
        self.przz_power = Some(Self::validate_positive(power, "RZZ power parameter"));
        self
    }

    /// Add a gate type to the set of noiseless gates
    #[must_use]
    pub fn with_noiseless_gate(mut self, gate_type: GateType) -> Self {
        if self.noiseless_gates.is_none() {
            self.noiseless_gates = Some(HashSet::new());
        }

        if let Some(ref mut gates) = self.noiseless_gates {
            gates.insert(gate_type);
        }

        self
    }

    /// Set whether to replace leakage with depolarizing noise
    #[must_use]
    pub fn with_leak2depolar(mut self, use_depolar: bool) -> Self {
        self.leak2depolar = Some(use_depolar);
        self
    }

    /// Set the scaling factor for measurement crosstalk probability
    ///
    /// Additional scaling factor specifically for measurement crosstalk probability.
    #[must_use]
    pub fn with_p_meas_crosstalk_scale(mut self, scale: f64) -> Self {
        self.p_meas_crosstalk_scale = Some(Self::validate_non_negative(
            scale,
            "Measurement crosstalk rescale factor",
        ));
        self
    }

    /// Set the scaling factor for initialization crosstalk probability
    ///
    /// Additional scaling factor specifically for initialization crosstalk probability.
    #[must_use]
    pub fn with_p_prep_crosstalk_scale(mut self, scale: f64) -> Self {
        self.p_prep_crosstalk_scale = Some(Self::validate_non_negative(
            scale,
            "Preparation crosstalk rescale factor",
        ));
        self
    }

    /// Set the probability model for two-qubit Pauli errors
    #[must_use]
    pub fn with_p2_pauli_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        self.p2_pauli_model = Some(TwoQubitWeightedSampler::new(model));
        self
    }

    /// Set the probability model for two-qubit emission errors
    #[must_use]
    pub fn with_p2_emission_model(mut self, model: &BTreeMap<String, f64>) -> Self {
        self.p2_emission_model = Some(TwoQubitWeightedSampler::new(model));
        self
    }

    /// Set the emission ratio for single-qubit gate errors
    #[must_use]
    pub fn with_p1_emission_ratio(mut self, ratio: f64) -> Self {
        self.p1_emission_ratio = Some(Self::validate_probability(ratio));
        self
    }

    /// Set the two-qubit emission ratio
    #[must_use]
    pub fn with_p2_emission_ratio(mut self, ratio: f64) -> Self {
        self.p2_emission_ratio = Some(Self::validate_probability(ratio));
        self
    }

    /// Set the probability of a leaked qubit being seeped (released from leakage)
    #[must_use]
    pub fn with_p1_seepage_prob(mut self, prob: f64) -> Self {
        self.p1_seepage_prob = Some(Self::validate_probability(prob));
        self
    }

    /// Set the probability of a leaked qubit being seeped (released from leakage)
    #[must_use]
    pub fn with_p2_seepage_prob(mut self, prob: f64) -> Self {
        self.p2_seepage_prob = Some(Self::validate_probability(prob));
        self
    }

    /// Set the probability of a leaked qubit being seeped (released from leakage)
    #[must_use]
    pub fn with_seepage_prob(mut self, prob: f64) -> Self {
        self.p1_seepage_prob = Some(Self::validate_probability(prob));
        self.p2_seepage_prob = Some(Self::validate_probability(prob));
        self
    }

    /// Set the probability of crosstalk during measurement operations
    #[must_use]
    pub fn with_p_crosstalk_meas(mut self, prob: f64) -> Self {
        self.p_meas_crosstalk = Some(Self::validate_probability(prob));
        self
    }

    /// Set the probability of crosstalk during initialization operations
    #[must_use]
    pub fn with_p_crosstalk_prep(mut self, prob: f64) -> Self {
        self.p_prep_crosstalk = Some(Self::validate_probability(prob));
        self
    }

    /// Set whether to apply crosstalk on a per-gate basis
    #[must_use]
    pub fn with_crosstalk_per_gate(mut self, per_gate: bool) -> Self {
        self.crosstalk_per_gate = Some(per_gate);
        self
    }

    /// Scale error probabilities based on scaling factors
    ///
    /// This method applies all scaling factors to the error probabilities:
    /// - Global scale factor
    /// - Type-specific scale factors (measurement, preparation, memory, etc.)
    /// - Conversion factors from average to total error rates (3/2 for p1, 5/4 for p2)
    ///
    /// This method should be called exactly once after setting all parameters
    /// and before using the noise model for simulation. Calling it multiple times will
    /// compound the scaling factors incorrectly.
    pub fn scale_parameters(&mut self, model: &mut GeneralNoiseModel) {
        let scale = self.scale.unwrap_or(1.0);
        // let memory_scale = self.memory_scale.unwrap_or(1.0);
        let prep_scale = self.prep_scale.unwrap_or(1.0);
        let meas_scale = self.meas_scale.unwrap_or(1.0);
        let leakage_scale = self.leakage_scale.unwrap_or(1.0);
        let p1_scale = self.p1_scale.unwrap_or(1.0);
        let p2_scale = self.p2_scale.unwrap_or(1.0);
        let emission_scale = self.emission_scale.unwrap_or(1.0);
        let p_meas_crosstalk_scale = self.p_meas_crosstalk_scale.unwrap_or(1.0);
        let p_prep_crosstalk_scale = self.p_prep_crosstalk_scale.unwrap_or(1.0);

        // Apply dephasing errors based on the duration
        // Use memory_scale to adjust the dephasing rate
        // model.dephasing_rate *= self.memory_scale * self.scale;

        // Scale single-qubit gate error probability
        model.p1 *= p1_scale * scale;

        // Scale two-qubit gate error probability
        model.p2 *= p2_scale * scale;

        model.p_meas_0 *= meas_scale * scale;
        model.p_meas_1 *= meas_scale * scale;

        // Scale preparation error probability
        model.p_prep *= prep_scale * scale;

        // Scale preparation leakage ratio - include the global scale factor
        model.p_prep_leak_ratio *= leakage_scale * scale;
        model.p_prep_leak_ratio = model.p_prep_leak_ratio.min(1.0);

        // Apply crosstalk rescaling factors
        model.p_meas_crosstalk *= p_meas_crosstalk_scale;
        model.p_prep_crosstalk *= p_prep_crosstalk_scale;

        // Then apply the regular scaling to crosstalks
        model.p_meas_crosstalk *= meas_scale * scale;
        model.p_prep_crosstalk *= prep_scale * scale;

        // Scale emission ratios
        model.p1_emission_ratio *= emission_scale * scale;
        model.p1_emission_ratio = model.p1_emission_ratio.min(1.0);

        model.p2_emission_ratio *= emission_scale * scale;
        model.p2_emission_ratio = model.p2_emission_ratio.min(1.0);
    }

    /// Build the general noise model
    ///
    /// TODO: Consider another build with noiseless default
    ///
    /// # Returns
    /// A boxed noise model
    ///
    /// # Panics
    /// Panics if any probabilities are not set or are not between 0 and 1.
    #[must_use]
    pub fn build(mut self) -> Box<dyn NoiseModel> {
        // Start with the default noise model as a base
        let mut model = GeneralNoiseModel::default();

        // Apply all parameters that were explicitly set
        if let Some(p_prep) = self.p_prep {
            model.p_prep = p_prep;
        }

        if let Some(p_meas_0) = self.p_meas_0 {
            model.p_meas_0 = p_meas_0;
        }

        if let Some(p_meas_1) = self.p_meas_1 {
            model.p_meas_1 = p_meas_1;
        }

        model.p_meas_max = model.p_meas_0.max(model.p_meas_1);

        if let Some(p1) = self.p1 {
            model.p1 = p1;
        }

        if let Some(p2) = self.p2 {
            model.p2 = p2;
        }

        if let Some(ratio) = self.p1_emission_ratio {
            model.p1_emission_ratio = ratio;
        }

        if let Some(ratio) = self.p2_emission_ratio {
            model.p2_emission_ratio = ratio;
        }

        if let Some(model_map) = self.p1_pauli_model.clone() {
            model.p1_pauli_model = model_map;
        }

        if let Some(model_map) = self.p1_emission_model.clone() {
            model.p1_emission_model = model_map;
        }

        if let Some(model_map) = self.p2_pauli_model.clone() {
            model.p2_pauli_model = model_map;
        }

        if let Some(model_map) = self.p2_emission_model.clone() {
            model.p2_emission_model = model_map;
        }

        if let Some(ratio) = self.p_prep_leak_ratio {
            model.p_prep_leak_ratio = ratio;
        }

        if let Some(prob) = self.p1_seepage_prob {
            model.p1_seepage_prob = prob;
        }

        if let Some(prob) = self.p2_seepage_prob {
            model.p2_seepage_prob = prob;
        }

        if let Some(seed) = self.seed {
            // Use the with_seed constructor for NoiseRng
            model.rng = NoiseRng::with_seed(seed);
        }

        if let Some(coherent) = self.coherent_dephasing {
            model.coherent_dephasing = coherent;
        }

        if let Some(factor) = self.coherent_to_incoherent_factor {
            model.coherent_to_incoherent_factor = factor;
        }

        if let Some(przz_params) = self.przz_params {
            model.przz_a = przz_params.0;
            model.przz_b = przz_params.1;
            model.przz_c = przz_params.2;
            model.przz_d = przz_params.3;
        }

        if let Some(power) = self.przz_power {
            model.przz_power = power;
        }

        if let Some(gates) = self.noiseless_gates.clone() {
            for gate in gates {
                model.add_noiseless_gate(gate);
            }
        }

        if let Some(leak2depolar) = self.leak2depolar {
            model.leak2depolar = leak2depolar;
        }

        if let Some(has_crosstalk_per_gate) = self.crosstalk_per_gate {
            model.crosstalk_per_gate = has_crosstalk_per_gate;
        }

        if let Some(prob) = self.p_meas_crosstalk {
            model.p_meas_crosstalk = prob;
        }

        if let Some(prob) = self.p_prep_crosstalk {
            model.p_prep_crosstalk = prob;
        }

        self.scale_parameters(&mut model);
        Box::new(model)
    }

    /// Create a new builder from an existing model's configuration
    ///
    /// This method is useful for creating a new model that is identical to an existing one
    /// except for a few changed parameters.
    ///
    /// # Arguments
    /// * `model` - The existing model to copy parameters from
    ///
    /// # Returns
    /// A builder with parameters copied from the existing model
    #[must_use]
    pub fn from_model(model: &GeneralNoiseModel) -> Self {
        Self {
            p_prep: Some(model.p_prep),
            p_meas_0: Some(model.p_meas_0),
            p_meas_1: Some(model.p_meas_1),
            p1: Some(model.p1),
            p2: Some(model.p2),
            p1_emission_ratio: Some(model.p1_emission_ratio),
            p2_emission_ratio: Some(model.p2_emission_ratio),
            p1_pauli_model: Some(model.p1_pauli_model.clone()),
            p1_emission_model: Some(model.p1_emission_model.clone()),
            p2_pauli_model: Some(model.p2_pauli_model.clone()),
            p2_emission_model: Some(model.p2_emission_model.clone()),
            p_prep_leak_ratio: Some(model.p_prep_leak_ratio),
            p1_seepage_prob: Some(model.p1_seepage_prob),
            p2_seepage_prob: Some(model.p2_seepage_prob),
            seed: None, // Don't copy the seed
            scale: None,
            memory_scale: None,
            prep_scale: None,
            meas_scale: None,
            leakage_scale: None,
            p1_scale: None,
            p2_scale: None,
            emission_scale: None,
            p_meas_crosstalk: Some(model.p_meas_crosstalk),
            p_prep_crosstalk: Some(model.p_prep_crosstalk),
            p_meas_crosstalk_scale: None,
            p_prep_crosstalk_scale: None,
            crosstalk_per_gate: Some(model.crosstalk_per_gate),
            coherent_dephasing: Some(model.coherent_dephasing),
            coherent_to_incoherent_factor: Some(model.coherent_to_incoherent_factor),
            przz_params: Some((model.przz_a, model.przz_b, model.przz_c, model.przz_d)),
            przz_power: Some(model.przz_power),
            noiseless_gates: Some(model.noiseless_gates.clone()),
            leak2depolar: Some(model.leak2depolar),
        }
    }
}

impl Default for GeneralNoiseModel {
    /// Create a new noise model with default error parameters
    ///
    /// Creates a `GeneralNoiseModel` with sensible default error probabilities:
    /// * `p_prep` - Preparation (initialization) error probability: 0.01
    /// * `p_meas_0` - Probability of measuring 1 when the state is |0⟩: 0.01
    /// * `p_meas_1` - Probability of measuring 0 when the state is |1⟩: 0.01
    /// * `p1` - Single-qubit gate error probability (average error rate): 0.001
    /// * `p2` - Two-qubit gate error probability (average error rate): 0.01
    ///
    /// Other parameters are initialized with sensible defaults, including uniform
    /// distributions for Pauli errors and emission errors.
    ///
    /// # Example
    /// ```
    /// use pecos_engines::engines::noise::GeneralNoiseModel;
    ///
    /// // Create model with default error probabilities
    /// let mut model = GeneralNoiseModel::default();
    /// ```
    fn default() -> Self {
        // Initialize default models
        let mut p1_pauli_model = BTreeMap::new();
        p1_pauli_model.insert("X".to_string(), 1.0 / 3.0);
        p1_pauli_model.insert("Y".to_string(), 1.0 / 3.0);
        p1_pauli_model.insert("Z".to_string(), 1.0 / 3.0);

        let mut p1_emission_model = BTreeMap::new();
        p1_emission_model.insert("X".to_string(), 1.0 / 3.0);
        p1_emission_model.insert("Y".to_string(), 1.0 / 3.0);
        p1_emission_model.insert("Z".to_string(), 1.0 / 3.0);

        let mut p2_pauli_model = BTreeMap::new();
        p2_pauli_model.insert("XX".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("XY".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("XZ".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("YX".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("YY".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("YZ".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("ZX".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("ZY".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("ZZ".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("IX".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("IY".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("IZ".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("XI".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("YI".to_string(), 1.0 / 15.0);
        p2_pauli_model.insert("ZI".to_string(), 1.0 / 15.0);

        let mut p2_emission_model = BTreeMap::new();
        p2_emission_model.insert("XX".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("XY".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("XZ".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("YX".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("YY".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("YZ".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("ZX".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("ZY".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("ZZ".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("IX".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("IY".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("IZ".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("XI".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("YI".to_string(), 1.0 / 15.0);
        p2_emission_model.insert("ZI".to_string(), 1.0 / 15.0);

        let p_meas_0: f64 = 0.01; // 1% probability of measuring 1 when state is |0⟩
        let p_meas_1: f64 = 0.01; // 1% probability of measuring 0 when state is |1⟩

        // Default error probabilities
        Self {
            p_prep: 0.01,
            p_meas_0,
            p_meas_1,
            p1: 0.001,
            p2: 0.01,
            p1_emission_ratio: 0.5,
            p_prep_leak_ratio: 0.5,
            p2_emission_ratio: 0.5,
            p1_pauli_model: SingleQubitWeightedSampler::new(&p1_pauli_model),
            p1_emission_model: SingleQubitWeightedSampler::new(&p1_emission_model),
            p2_pauli_model: TwoQubitWeightedSampler::new(&p2_pauli_model),
            p2_emission_model: TwoQubitWeightedSampler::new(&p2_emission_model),
            p1_seepage_prob: 0.5,
            p2_seepage_prob: 0.5,
            przz_a: 0.0,
            przz_b: 1.0,
            przz_c: 0.0,
            przz_d: 1.0,
            przz_power: 1.0,
            leaked_qubits: HashSet::new(),
            rng: NoiseRng::default(),
            p_meas_crosstalk: 0.0,
            p_prep_crosstalk: 0.0,
            crosstalk_per_gate: false,
            coherent_dephasing: false,
            coherent_to_incoherent_factor: 2.0,
            noiseless_gates: HashSet::new(),
            p_meas_max: p_meas_0.max(p_meas_1),
            leak2depolar: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::byte_message::ByteMessageBuilder;
    use crate::byte_message::gate_type::{GateType, QuantumGate};

    #[test]
    fn test_default() {
        // Create a noise model with the default settings
        let model = GeneralNoiseModel::default();

        // Check the default values
        assert!(
            (model.p_prep - 0.01).abs() < f64::EPSILON,
            "Default p_prep should be 0.01"
        );
        assert!(
            (model.p_meas_0 - 0.01).abs() < f64::EPSILON,
            "Default p_meas_0 should be 0.01"
        );
        assert!(
            (model.p_meas_1 - 0.01).abs() < f64::EPSILON,
            "Default p_meas_1 should be 0.01"
        );
        assert!(
            (model.p1 - 0.001).abs() < f64::EPSILON,
            "Default p1 should be 0.001"
        );
        assert!(
            (model.p2 - 0.01).abs() < f64::EPSILON,
            "Default p2 should be 0.01"
        );
        assert!(
            (model.p1_emission_ratio - 0.5).abs() < f64::EPSILON,
            "Default p1_emission_ratio should be 0.5"
        );
        assert!(
            (model.p_prep_leak_ratio - 0.5).abs() < f64::EPSILON,
            "Default p_prep_leak_ratio should be 0.5"
        );
        assert!(
            (model.p2_emission_ratio - 0.5).abs() < f64::EPSILON,
            "Default p2_emission_ratio should be 0.5"
        );
        assert!(
            (model.p1_seepage_prob - 0.5).abs() < f64::EPSILON,
            "Default seepage_prob should be 0.5"
        );
        assert!(
            (model.p2_seepage_prob - 0.5).abs() < f64::EPSILON,
            "Default seepage_prob should be 0.5"
        );
    }

    #[test]
    fn test_builder() {
        // Create a noise model with the builder
        let noise = GeneralNoiseModel::builder()
            .with_prep_probability(0.1)
            .with_meas_0_probability(0.2)
            .with_meas_1_probability(0.3)
            .with_average_p1_probability(0.4)
            .with_average_p2_probability(0.5)
            .with_prep_leak_ratio(0.6)
            .build();

        // Get the boxed noise model's probabilities using any_ref downcast
        let noise_ref = noise.as_any().downcast_ref::<GeneralNoiseModel>().unwrap();
        let (p_prep, p_meas_0, p_meas_1, p1, p2, p_prep_leak_ratio) = noise_ref.probabilities();

        // Print the actual values - with proper scaling, p1 and p2 should include the 3/2 and 5/4 factors
        let expected_p1 = 0.4 * (3.0 / 2.0);
        let expected_p2 = 0.5 * (5.0 / 4.0);

        println!(
            "Builder test: p1 actual: {}, expected: {}, diff: {}",
            p1,
            expected_p1,
            (p1 - expected_p1).abs()
        );
        println!(
            "Builder test: p2 actual: {}, expected: {}, diff: {}",
            p2,
            expected_p2,
            (p2 - expected_p2).abs()
        );

        // Check the values
        assert!((p_prep - 0.1).abs() < f64::EPSILON);
        assert!((p_meas_0 - 0.2).abs() < f64::EPSILON);
        assert!((p_meas_1 - 0.3).abs() < f64::EPSILON);

        // With proper scaling, p1 and p2 should include just one application of the 3/2 and 5/4 factors
        let epsilon = 1e-6;
        assert!(
            (p1 - expected_p1).abs() < epsilon,
            "p1 mismatch: actual={}, expected={}, diff={}",
            p1,
            expected_p1,
            (p1 - expected_p1).abs()
        );

        assert!(
            (p2 - expected_p2).abs() < epsilon,
            "p2 mismatch: actual={}, expected={}, diff={}",
            p2,
            expected_p2,
            (p2 - expected_p2).abs()
        );

        assert!((p_prep_leak_ratio - 0.6).abs() < f64::EPSILON);

        // Test the builder with no parameters (should use defaults)
        let default_noise = GeneralNoiseModel::builder().build();
        let default_ref = default_noise
            .as_any()
            .downcast_ref::<GeneralNoiseModel>()
            .unwrap();

        // Verify a few key default values
        assert!(
            (default_ref.p1 - 0.001).abs() < 1e-6,
            "Default p1 should be 0.001"
        );
        assert!(
            (default_ref.p2 - 0.01).abs() < 1e-6,
            "Default p2 should be 0.01"
        );
    }

    #[test]
    fn test_biased_measurement() {
        use crate::byte_message::ByteMessageBuilder;

        // Create a noise model with 100% flip probabilities for deterministic testing
        let mut noise = GeneralNoiseModel::new(0.0, 1.0, 1.0, 0.0, 0.0);

        // Create a message with a 0 measurement result
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_measurement_results();
        builder.add_measurement_results(&[0], &[0]);
        let message_with_zero = builder.build();

        // Test measurement bias - all 0s should be flipped to 1s
        let biased_zero = noise
            .apply_noise_on_continue_processing(message_with_zero)
            .unwrap();
        let results = biased_zero.measurement_results_as_vec().unwrap();
        assert_eq!(results[0].1, 1, "0 should be flipped to 1");

        // Create a message with a 1 measurement result
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_measurement_results();
        builder.add_measurement_results(&[1], &[0]);
        let message_with_one = builder.build();

        // Test measurement bias - all 1s should be flipped to 0s
        let biased_one = noise
            .apply_noise_on_continue_processing(message_with_one)
            .unwrap();
        let results = biased_one.measurement_results_as_vec().unwrap();
        assert_eq!(results[0].1, 0, "1 should be flipped to 0");

        // Create a noise model with 0% flip probabilities
        noise = GeneralNoiseModel::new(0.0, 0.0, 0.0, 0.0, 0.0);

        // Test measurement bias with 0% flip - all 0s should remain 0s
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_measurement_results();
        builder.add_measurement_results(&[0], &[0]);
        let message_with_zero = builder.build();

        let unbiased_zero = noise
            .apply_noise_on_continue_processing(message_with_zero)
            .unwrap();
        let results = unbiased_zero.measurement_results_as_vec().unwrap();
        assert_eq!(results[0].1, 0, "0 should remain 0");

        // Test measurement bias with 0% flip - all 1s should remain 1s
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_measurement_results();
        builder.add_measurement_results(&[1], &[0]);
        let message_with_one = builder.build();

        let unbiased_one = noise
            .apply_noise_on_continue_processing(message_with_one)
            .unwrap();
        let results = unbiased_one.measurement_results_as_vec().unwrap();
        assert_eq!(results[0].1, 1, "1 should remain 1");
    }

    #[test]
    fn test_prep_leak_ratio() {
        use crate::byte_message::{ByteMessageBuilder, GateType, QuantumGate};

        // Create a noise model with 100% prep error probability and 100% leakage ratio
        // using the builder pattern
        let mut model = GeneralNoiseModel::builder()
            .with_prep_probability(1.0)
            .with_prep_leak_ratio(1.0)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Create a quantum gate operation (Prep on qubit 0)
        let gate = QuantumGate {
            gate_type: GateType::Prep,
            qubits: vec![0],
            params: vec![],
            result_id: None,
            noiseless: false,
        };

        // Create a builder and apply noise
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Add the gate and apply prep faults - this should cause leakage
        noise.apply_prep_faults(&gate, &mut builder);

        // Verify qubit 0 is now leaked
        assert!(noise.is_leaked(0), "Qubit 0 should be marked as leaked");

        // Now, create a noise model with 100% prep error probability but 0% leakage ratio
        let mut model = GeneralNoiseModel::builder()
            .with_prep_probability(1.0)
            .with_prep_leak_ratio(0.0)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Create a new builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Apply noise - this should NOT cause leakage
        noise.apply_prep_faults(&gate, &mut builder);

        // Verify qubit 0 is not leaked
        assert!(
            !noise.is_leaked(0),
            "Qubit 0 should not be marked as leaked"
        );

        // Test builder configuration
        let noise = GeneralNoiseModel::builder()
            .with_prep_probability(0.1)
            .with_meas_0_probability(0.1)
            .with_meas_1_probability(0.1)
            .with_p1_probability(0.1)
            .with_p2_probability(0.1)
            .with_prep_leak_ratio(0.7)
            .build();

        // Verify the prep leak ratio is set correctly
        let noise_ref = noise.as_any().downcast_ref::<GeneralNoiseModel>().unwrap();
        let (_, _, _, _, _, p_prep_leak_ratio) = noise_ref.probabilities();
        assert!(
            (p_prep_leak_ratio - 0.7).abs() < f64::EPSILON,
            "Prep leak ratio should be 0.7"
        );
    }

    #[test]
    fn test_leaked_qubit_measurement_behavior() {
        use crate::byte_message::ByteMessageBuilder;

        // Create a noise model with no spontaneous errors
        let mut model = GeneralNoiseModel::builder()
            .with_prep_probability(0.0)
            .with_meas_0_probability(0.0)
            .with_meas_1_probability(0.0)
            .with_p1_probability(0.0)
            .with_p2_probability(0.0)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Manually mark qubit 0 as leaked
        noise.mark_as_leaked(0);

        // In the test, we want to verify that the qubit becomes unleaked after measurement
        // Create a forced result message - this simulates the correct behavior
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_measurement_results();
        builder.add_measurement_results(&[1], &[0]); // Force outcome to 1 for result_id 0

        // "Apply bias" - in practice this will just check for and unleaked any measured qubits
        let biased_message = noise
            .apply_noise_on_continue_processing(builder.build())
            .unwrap();

        // Get the measurement results
        let results = biased_message.measurement_results_as_vec().unwrap();

        // Verify that the leaked qubit is reported as measured as 1
        assert_eq!(results[0].1, 1, "Leaked qubit should always measure as 1");

        // Verify that the qubit is no longer leaked after measurement
        // This is what we really care about testing - the leaked state cleanup
        assert!(
            !noise.is_leaked(0),
            "Qubit should be unleaked after measurement"
        );
    }

    #[test]
    fn test_parameter_scaling() {
        // Test that scaling factors are applied correctly - use builder pattern
        let mut model = GeneralNoiseModel::builder()
            .with_prep_probability(0.01)
            .with_meas_0_probability(0.01)
            .with_meas_1_probability(0.01)
            .with_average_p1_probability(0.01)
            .with_average_p2_probability(0.01)
            .with_scale(2.0)
            .with_p1_scale(3.0)
            .with_p2_scale(4.0)
            .with_prep_scale(5.0)
            .with_meas_scale(6.0)
            .with_leakage_scale(0.25)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Get values after scaling
        let (p_prep, p_meas_0, p_meas_1, p1, p2, p_prep_leak_ratio) = noise.probabilities();

        // Compare with expected values
        let expected_p_prep = 0.01 * 5.0 * 2.0; // Base * prep_scale * overall scale
        let expected_p_meas = 0.01 * 6.0 * 2.0; // Base * meas_scale * overall scale
        let expected_p1 = 0.01 * 3.0 * 2.0 * (3.0 / 2.0); // Base * p1_scale * overall scale * avg->total
        let expected_p2 = 0.01 * 4.0 * 2.0 * (5.0 / 4.0); // Base * p2_scale * overall scale * avg->total

        // Initial value in constructor is 0.5
        // and we scale it by leakage_scale (0.25) and overall scale (2.0)
        let expected_leak_ratio = 0.5 * 0.25 * 2.0; // Base * leakage_scale * overall scale, capped at 1.0

        println!(
            "p1 actual: {}, expected: {}, diff: {}",
            p1,
            expected_p1,
            (p1 - expected_p1).abs()
        );
        println!(
            "p2 actual: {}, expected: {}, diff: {}",
            p2,
            expected_p2,
            (p2 - expected_p2).abs()
        );

        // Check scaled values with a small epsilon for floating point comparison
        assert!(
            (p_prep - expected_p_prep).abs() < 1e-6,
            "p_prep mismatch: actual={p_prep}, expected={expected_p_prep}"
        );
        assert!(
            (p_meas_0 - expected_p_meas).abs() < 1e-6,
            "p_meas_0 mismatch: actual={p_meas_0}, expected={expected_p_meas}"
        );
        assert!(
            (p_meas_1 - expected_p_meas).abs() < 1e-6,
            "p_meas_1 mismatch: actual={p_meas_1}, expected={expected_p_meas}"
        );
        assert!(
            (p1 - expected_p1).abs() < 1e-6,
            "p1 mismatch: actual={p1}, expected={expected_p1}"
        );
        assert!(
            (p2 - expected_p2).abs() < 1e-6,
            "p2 mismatch: actual={p2}, expected={expected_p2}"
        );
        assert!(
            (p_prep_leak_ratio - expected_leak_ratio).abs() < 1e-6,
            "p_prep_leak_ratio mismatch: actual={p_prep_leak_ratio}, expected={expected_leak_ratio}"
        );
    }

    #[test]
    fn test_builder_with_scaling() {
        // Test that builder applies scaling factors correctly
        let noise = GeneralNoiseModel::builder()
            .with_prep_probability(0.01)
            .with_meas_0_probability(0.01)
            .with_meas_1_probability(0.01)
            .with_average_p1_probability(0.01)
            .with_average_p2_probability(0.01)
            .with_prep_leak_ratio(0.01)
            .with_scale(2.0)
            .with_p1_scale(3.0)
            .with_p2_scale(4.0)
            .with_prep_scale(5.0)
            .with_meas_scale(6.0)
            .with_leakage_scale(7.0)
            .build();

        // Downcast to check properties
        let noise_ref = noise.as_any().downcast_ref::<GeneralNoiseModel>().unwrap();
        let (p_prep, p_meas_0, p_meas_1, p1, p2, p_prep_leak_ratio) = noise_ref.probabilities();

        // With single scaling, values should match what we expect from applying
        // the scale_parameters method once
        let expected_p_prep = 0.01 * 5.0 * 2.0; // Base * prep_scale * overall scale
        let expected_p_meas = 0.01 * 6.0 * 2.0; // Base * meas_scale * overall scale
        let expected_p1 = 0.01 * 3.0 * 2.0 * (3.0 / 2.0); // Base * p1_scale * overall scale * avg->total
        let expected_p2 = 0.01 * 4.0 * 2.0 * (5.0 / 4.0); // Base * p2_scale * overall scale * avg->total

        // When using with_uniform_probability(0.01), p_prep_leak_ratio is set to 0.01
        // and we scale it by leakage_scale (7.0) and overall scale (2.0)
        let expected_leak_ratio = 0.01 * 7.0 * 2.0; // Base * leakage_scale * overall scale

        println!("Builder with scaling test:");
        println!(
            "p1 actual: {}, expected: {}, diff: {}",
            p1,
            expected_p1,
            (p1 - expected_p1).abs()
        );
        println!(
            "p2 actual: {}, expected: {}, diff: {}",
            p2,
            expected_p2,
            (p2 - expected_p2).abs()
        );

        // Check scaled values with a small epsilon for floating point comparison
        assert!(
            (p_prep - expected_p_prep).abs() < 1e-6,
            "p_prep mismatch: actual={p_prep}, expected={expected_p_prep}"
        );
        assert!(
            (p_meas_0 - expected_p_meas).abs() < 1e-6,
            "p_meas_0 mismatch: actual={p_meas_0}, expected={expected_p_meas}"
        );
        assert!(
            (p_meas_1 - expected_p_meas).abs() < 1e-6,
            "p_meas_1 mismatch: actual={p_meas_1}, expected={expected_p_meas}"
        );
        assert!(
            (p1 - expected_p1).abs() < 1e-6,
            "p1 mismatch: actual={p1}, expected={expected_p1}"
        );
        assert!(
            (p2 - expected_p2).abs() < 1e-6,
            "p2 mismatch: actual={p2}, expected={expected_p2}"
        );
        assert!(
            (p_prep_leak_ratio - expected_leak_ratio).abs() < 1e-6,
            "p_prep_leak_ratio mismatch: actual={p_prep_leak_ratio}, expected={expected_leak_ratio}"
        );
    }

    #[test]
    fn test_emission_ratio_scaling() {
        // Test that emission ratios are properly scaled and capped at a maximum of 1.0
        // Default emission ratios are 0.5
        let mut model = GeneralNoiseModel::builder()
            .with_scale(3.0)
            .with_emission_scale(4.0)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Verify both ratios are 0.5 after scaling
        // When scaled: 0.5 * 3.0 (scale) * 4.0 (emission_scale) = 6.0
        // But capped at 1.0
        assert!(
            (noise.p1_emission_ratio - 1.0).abs() < 1e-6,
            "p1_emission_ratio should be 1.0 after scaling/capping"
        );
        assert!(
            (noise.p2_emission_ratio - 1.0).abs() < 1e-6,
            "p2_emission_ratio should be 1.0 after scaling/capping"
        );

        // Now test with values that won't exceed the cap
        let mut model = GeneralNoiseModel::builder()
            .with_p1_emission_ratio(0.1)
            .with_p2_emission_ratio(0.1)
            .with_scale(2.0)
            .with_emission_scale(3.0)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Expected values: 0.1 * 3.0 (emission) * 2.0 (overall) = 0.6
        assert!((noise.p1_emission_ratio - 0.6).abs() < 1e-6);
        assert!((noise.p2_emission_ratio - 0.6).abs() < 1e-6);
    }

    // #[test]
    // fn test_coherent_dephasing() {
    //     // Create a circuit builder
    //     let mut builder = ByteMessageBuilder::new();
    //     let _ = builder.for_quantum_operations();
    //
    //     // Create a noise model with coherent dephasing
    //     let mut model = GeneralNoiseModel::builder()
    //         .with_coherent_dephasing(true)
    //         .build();
    //     let noise = model
    //         .as_any_mut()
    //         .downcast_mut::<GeneralNoiseModel>()
    //         .unwrap();
    //
    //     // Create an idle gate
    //     let gate = QuantumGate {
    //         gate_type: GateType::Idle,
    //         qubits: vec![0],
    //         params: vec![1.0], // 1 second duration
    //         result_id: None,
    //         noiseless: false,
    //     };
    //
    //     // Apply idle faults - should use coherent dephasing (RZ gates)
    //     noise.apply_idle_faults(&gate, &mut builder);
    //
    //     // Get the message and verify it contains RZ gates
    //     let message = builder.build();
    //     let gates = message.parse_quantum_operations().unwrap();
    //
    //     // At least one gate should be an RZ gate
    //     assert!(!gates.is_empty(), "Should have at least one gate");
    //     assert!(
    //         gates.iter().any(|g| g.gate_type == GateType::RZ),
    //         "Should contain at least one RZ gate"
    //     );
    //
    //     // Now test with incoherent dephasing
    //     let mut builder = ByteMessageBuilder::new();
    //     let _ = builder.for_quantum_operations();
    //
    //     let mut model = GeneralNoiseModel::builder()
    //         .with_coherent_dephasing(false)
    //         .with_seed(42)
    //         .build();
    //     let noise = model
    //         .as_any_mut()
    //         .downcast_mut::<GeneralNoiseModel>()
    //         .unwrap();
    //
    //     // Apply idle faults with incoherent dephasing
    //     noise.apply_idle_faults(&gate, &mut builder);
    //
    //     // The message may contain Z gates or be empty depending on random outcomes
    //     let message = builder.build();
    //     let _gates = message.parse_quantum_operations().unwrap();
    //
    //     // We can't assert specific outcomes due to randomness, but the code should run without errors
    // }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_rzz_error_rate() {
        let mut model = GeneralNoiseModel::builder()
            .with_average_p2_probability(0.1)
            .with_przz_params(0.1, 0.0, 0.25, 0.0)
            .with_przz_power(1.0)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Test negative angle
        let neg_theta = -std::f64::consts::PI / 2.0;
        let error_neg = noise.rzz_error_rate(neg_theta);
        let expected_neg = 0.00625;
        assert!(
            (error_neg - expected_neg).abs() < 1e-6,
            "Expected {expected_neg}, got {error_neg}"
        );

        // Test positive angle
        let pos_theta = std::f64::consts::PI / 2.0;
        let error_pos = noise.rzz_error_rate(pos_theta);
        let expected_pos = 0.015625;
        assert!(
            (error_pos - expected_pos).abs() < 1e-6,
            "Expected {expected_pos}, got {error_pos}"
        );

        // Test quadratic scaling
        let mut model = GeneralNoiseModel::builder()
            .with_average_p2_probability(0.1)
            .with_przz_params(0.1, 0.0, 0.25, 0.0)
            .with_przz_power(2.0)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        let error_quad = noise.rzz_error_rate(pos_theta);
        let expected_quad = 0.0078125;
        assert!(
            (error_quad - expected_quad).abs() < 1e-6,
            "Expected {expected_quad}, got {error_quad}"
        );
    }

    #[test]
    fn test_noiseless_gates() {
        // Create a noise model and mark RZ as a noiseless gate
        let mut model = GeneralNoiseModel::builder()
            .with_p1_probability(0.5) // Use a moderate valid probability
            .with_noiseless_gate(GateType::RZ)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Create a builder to capture gates
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Create an RZ gate (noiseless - should not have noise applied)
        let rz_gate = QuantumGate {
            gate_type: GateType::RZ,
            qubits: vec![0],
            params: vec![0.1],
            result_id: None,
            noiseless: false,
        };

        // Create an X gate (not noiseless - should have noise applied)
        let x_gate = QuantumGate {
            gate_type: GateType::X,
            qubits: vec![0],
            params: vec![],
            result_id: None,
            noiseless: false,
        };

        // Make sure RZ is recognized as noiseless
        assert!(
            noise.is_noiseless_gate(&GateType::RZ),
            "RZ should be a noiseless gate"
        );
        assert!(
            !noise.is_noiseless_gate(&GateType::X),
            "X should not be a noiseless gate"
        );

        let msg =
            ByteMessage::create_circuit_from_quantum_gates(&[rz_gate.clone(), x_gate.clone()])
                .expect("Something when wrong in the construction of a circuit");

        // Apply noise to the gates manually since we can't access apply_noise_to_gates directly
        let message = noise.apply_noise_on_start(&msg).unwrap();
        let gates = message.parse_quantum_operations().unwrap();

        // We expect the RZ gate to be unchanged, and the X gate might have errors applied
        // (can't verify exact count due to randomness, but we know we should have at least one)
        assert!(!gates.is_empty(), "Should have at least one gate");

        // We can verify the first gate is the RZ gate (unchanged)
        assert_eq!(
            gates[0].gate_type,
            GateType::RZ,
            "First gate should be RZ (unchanged)"
        );
    }

    #[test]
    fn test_leak2depolar() {
        // Create a noise model with leak2depolar set to true
        let mut model = GeneralNoiseModel::builder().with_leak2depolar(true).build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Create a builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Apply leak which should now use depolarizing errors instead
        noise.leak(0);

        // Verify the qubit is not marked as leaked
        assert!(
            !noise.is_leaked(0),
            "Qubit should not be marked as leaked with leak2depolar=true"
        );

        // Reset and try with leak2depolar=false
        let mut model = GeneralNoiseModel::builder()
            .with_leak2depolar(false)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Clear the builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Apply leak which should now mark the qubit as leaked
        noise.leak(0);

        // Verify the qubit is marked as leaked
        assert!(
            noise.is_leaked(0),
            "Qubit should be marked as leaked with leak2depolar=false"
        );
    }

    #[test]
    fn test_rzz_error_rate_debug() {
        let mut model = GeneralNoiseModel::builder()
            .with_average_p2_probability(0.1)
            .with_przz_params(0.1, 0.0, 0.25, 0.0)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Check unscaled przz error rate
        let theta = std::f64::consts::PI / 4.0;
        let norm_theta = theta / std::f64::consts::PI;
        let error_unscaled = noise.rzz_error_rate(theta);
        let c = 0.25;

        // After build(), parameters are scaled: p2 is scaled by 5/4
        let p2_scaled = 0.1 * (5.0 / 4.0);
        let expected_unscaled = c * norm_theta * p2_scaled; // 0.0078125

        assert!(
            (error_unscaled - expected_unscaled).abs() < 1e-6,
            "Expected {expected_unscaled}, got {error_unscaled}"
        );

        // Check scaled przz error rate
        let mut model = GeneralNoiseModel::builder()
            .with_average_p2_probability(0.1)
            .with_przz_params(0.1, 0.0, 0.25, 0.0)
            .with_scale(2.0)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        let error_scaled = noise.rzz_error_rate(theta);

        // After build() with scale 2.0, p2 is scaled by:
        // - scale (2.0)
        // - 5/4 conversion factor (from average to total error)
        let p2_scaled = 0.1 * 2.0 * (5.0 / 4.0);
        let expected_scaled = c * norm_theta * p2_scaled; // 0.015625

        assert!(
            (error_scaled - expected_scaled).abs() < 1e-6,
            "Expected {expected_scaled}, got {error_scaled}"
        );
    }

    #[test]
    fn test_pauli_and_emission_model_setters() {
        use std::collections::BTreeMap;
        // Define epsilon for approximate float comparisons
        const EPSILON: f64 = 0.005; // Increased tolerance for sampler discretization

        // Create all our custom models first
        let mut custom_p1_pauli = BTreeMap::new();
        custom_p1_pauli.insert("X".to_string(), 0.7);
        custom_p1_pauli.insert("Y".to_string(), 0.2);
        custom_p1_pauli.insert("Z".to_string(), 0.1);

        let mut custom_p1_emission = BTreeMap::new();
        custom_p1_emission.insert("X".to_string(), 0.4);
        custom_p1_emission.insert("Y".to_string(), 0.6);

        let mut custom_p2_pauli = BTreeMap::new();
        custom_p2_pauli.insert("XX".to_string(), 0.5);
        custom_p2_pauli.insert("YY".to_string(), 0.3);
        custom_p2_pauli.insert("ZZ".to_string(), 0.2);

        let mut custom_p2_emission = BTreeMap::new();
        custom_p2_emission.insert("XX".to_string(), 0.25);
        custom_p2_emission.insert("YY".to_string(), 0.75);

        // Create a noise model with custom Pauli and emission models using the builder
        let model = GeneralNoiseModel::builder()
            .with_prep_probability(0.01)
            .with_meas_0_probability(0.01)
            .with_meas_1_probability(0.01)
            .with_p1_probability(0.1)
            .with_p2_probability(0.2)
            .with_p1_pauli_model(&custom_p1_pauli)
            .with_p1_emission_model(&custom_p1_emission)
            .with_p2_pauli_model(&custom_p2_pauli)
            .with_p2_emission_model(&custom_p2_emission)
            .build();

        let noise = model.as_any().downcast_ref::<GeneralNoiseModel>().unwrap();

        // Get the distribution to verify using the direct accessor pattern
        let p1_pauli_dist = noise.p1_pauli_model().get_weighted_map();

        // Check that the distribution contains the right keys and approximate values
        assert!(
            p1_pauli_dist.contains_key("X"),
            "Distribution should contain X"
        );
        assert!(
            p1_pauli_dist.contains_key("Y"),
            "Distribution should contain Y"
        );
        assert!(
            p1_pauli_dist.contains_key("Z"),
            "Distribution should contain Z"
        );

        assert!(
            (p1_pauli_dist["X"] - 0.7).abs() < EPSILON,
            "Expected X value to be close to 0.7"
        );
        assert!(
            (p1_pauli_dist["Y"] - 0.2).abs() < EPSILON,
            "Expected Y value to be close to 0.2"
        );
        assert!(
            (p1_pauli_dist["Z"] - 0.1).abs() < EPSILON,
            "Expected Z value to be close to 0.1"
        );

        // Verify p1_emission_model was set correctly
        let p1_emission_dist = noise.p1_emission_model().get_weighted_map();
        assert!(
            p1_emission_dist.contains_key("X"),
            "Distribution should contain X"
        );
        assert!(
            p1_emission_dist.contains_key("Y"),
            "Distribution should contain Y"
        );

        assert!(
            (p1_emission_dist["X"] - 0.4).abs() < EPSILON,
            "Expected X value to be close to 0.4"
        );
        assert!(
            (p1_emission_dist["Y"] - 0.6).abs() < EPSILON,
            "Expected Y value to be close to 0.6"
        );

        // Verify p2_pauli_model was set correctly
        let p2_pauli_dist = noise.p2_pauli_model().get_weighted_map();
        assert!(
            p2_pauli_dist.contains_key("XX"),
            "Distribution should contain XX"
        );
        assert!(
            p2_pauli_dist.contains_key("YY"),
            "Distribution should contain YY"
        );
        assert!(
            p2_pauli_dist.contains_key("ZZ"),
            "Distribution should contain ZZ"
        );

        assert!(
            (p2_pauli_dist["XX"] - 0.5).abs() < EPSILON,
            "Expected XX value to be close to 0.5"
        );
        assert!(
            (p2_pauli_dist["YY"] - 0.3).abs() < EPSILON,
            "Expected YY value to be close to 0.3"
        );
        assert!(
            (p2_pauli_dist["ZZ"] - 0.2).abs() < EPSILON,
            "Expected ZZ value to be close to 0.2"
        );

        // Verify p2_emission_model was set correctly
        let p2_emission_dist = noise.p2_emission_model().get_weighted_map();
        assert!(
            p2_emission_dist.contains_key("XX"),
            "Distribution should contain XX"
        );
        assert!(
            p2_emission_dist.contains_key("YY"),
            "Distribution should contain YY"
        );

        assert!(
            (p2_emission_dist["XX"] - 0.25).abs() < EPSILON,
            "Expected XX value to be close to 0.25"
        );
        assert!(
            (p2_emission_dist["YY"] - 0.75).abs() < EPSILON,
            "Expected YY value to be close to 0.75"
        );
    }
}
