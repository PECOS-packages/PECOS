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
//! - Crosstalk errors between nearby qubits
//! - Memory/idle noise with T1/T2 processes
//! - Coherent vs. incoherent dephasing distinction
//!
//! Some features from the Python implementation that are not yet fully implemented:
//!
//! - Repumping cycles for leaked qubits
//! - Zone-specific error rates
//!
//! ## Usage
//!
//! The noise model can be instantiated directly or through a builder pattern:
//!
//! ```rust
//! use pecos_engines::noise::GeneralNoiseModel;
//!
//! // Using the builder with explicit error rates
//! let noise_model = GeneralNoiseModel::builder()
//!     .with_prep_probability(0.01)
//!     .with_meas_0_probability(0.02)
//!     .with_meas_1_probability(0.03)
//!     .with_p1_probability(0.04)
//!     .with_p2_probability(0.05)
//!     .with_seed(42)
//!     .build();
//! ```

#![allow(clippy::too_many_lines)]

mod builder;
mod default;

pub use self::builder::GeneralNoiseModelBuilder;

use crate::Gate;
use crate::byte_message::{ByteMessage, ByteMessageBuilder, GateType};
use crate::engine_system::{ControlEngine, EngineStage};
use crate::noise::noise_rng::NoiseRng;
use crate::noise::utils::NoiseUtils;
use crate::noise::utils::ProbabilityValidator;
use crate::noise::weighted_sampler::{
    CrosstalkWeightedSampler, SingleQubitWeightedSampler, TwoQubitWeightedSampler,
};
use crate::noise::{NoiseModel, RngManageable};
use log::trace;
use pecos_core::errors::PecosError;
use pecos_core::{Angle64, QubitId};
use pecos_random::PecosRng;
use std::any::Any;
use std::collections::BTreeSet;

/// General noise model with parameterized error channels.
///
/// Includes:
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
    noiseless_gates: BTreeSet<GateType>,

    /// Scale leakage events to be completely depolarizing events instead.
    ///
    /// 0.0 means no leakage and all leakage events are replaced with completely depolarizing noise.
    /// 1.0 means all leakage events remain leakage events.
    leakage_scale: f64,

    /// Whether to use coherent dephasing vs incoherent (stochastic) dephasing
    ///
    /// If true, dephasing is modeled as coherent phase rotations using RZ gates.
    /// If false, dephasing is modeled as stochastic Z errors with quadratic scaling.
    ///
    /// In physical systems, coherent dephasing represents systematic phase evolution
    /// such as frequency offsets.
    p_idle_coherent: bool,

    /// The idle noise rate for linear dependency on time (seconds).
    ///
    /// This always applies stochastic noise
    p_idle_linear_rate: f64,

    /// The stochastic model for the idle noise that has a linear dependency on time (seconds).
    ///
    /// Specifies the distribution of different stochastic idle noise types that can occur.
    ///
    /// The distribution is stored as pre-computed, cached sampler instead of the `HashMap` that is
    /// the input.
    p_idle_linear_model: SingleQubitWeightedSampler,

    /// The idle noise rate for quadratic dependency on time (seconds).
    ///
    /// This will be a coherent noise channel unless `p_idle_coherent` is set to false. If it is
    /// false it will apply Z to each qubit quadratic dependency on time
    p_idle_quadratic_rate: f64,

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
    p_idle_coherent_to_incoherent_factor: f64,

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

    /// Probability of crosstalk during preparation operations
    ///
    /// Models the probability that a preparation operation on one qubit affects nearby qubits.
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

    /// Probability model for emission errors on single qubit gates
    ///
    /// Specifies the distribution of different spontaneous emission error types that can occur.
    /// This includes errors that may cause state transitions outside the computational basis.
    ///
    /// The distribution is stored as pre-computed, cached sampler instead of the `HashMap` that is
    /// the input.
    p1_emission_model: SingleQubitWeightedSampler,

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
    /// The distribution is stored as pre-computed, cached sampler instead of the `HashMap` that is
    /// the input.
    p1_pauli_model: SingleQubitWeightedSampler,

    /// Probability of applying a fault after two-qubit gates
    ///
    /// Models depolarizing channel + leakage noise for two-qubit gates.
    p2: f64,

    /// Scaling parameters for two-qubit gate error rate - coefficient a
    ///
    /// Part of a parameterized model for angle-dependent errors in two-qubit gates.
    /// The error rate is modeled as a function of angle θ: p(θ) = a + b|θ| + c|θ|^d
    p2_angle_a: f64,

    /// Scaling parameters for two-qubit gate angular error rate dependency - coefficient b
    p2_angle_b: f64,

    /// Scaling parameters for two-qubit gate angular error rate dependency - coefficient c
    p2_angle_c: f64,

    /// Scaling parameters for two-qubit gate angular error rate dependency- coefficient d
    p2_angle_d: f64,

    /// Power parameter for two-qubit gate angular error rate dependency
    ///
    /// Controls how error probabilities scale with rotation angle in two-qubit gates.
    /// Error scales as `theta^p2_angle_power` where theta is the gate angle.
    /// Typically set to 1.0 for linear scaling.
    p2_angle_power: f64,

    /// The proportion of two-qubit errors that are emission faults
    ///
    /// Controls what fraction of faults on two-qubit gates are spontaneous emission faults versus
    /// standard depolarizing faults. In ion trap systems, this could model decay or transitions to
    /// non-computational states during two-qubit operations. Ranges from 0 to 1.
    p2_emission_ratio: f64,

    /// Probability model for spontaneous emission errors on two-qubit gates
    ///
    /// Specifies the distribution of different emission error types that can occur during
    /// two-qubit operations. This includes errors that may cause state transitions outside
    /// the computational basis.
    ///
    /// The distribution is stored as pre-computed, cached sampler instead of the `HashMap` that is the input.
    p2_emission_model: TwoQubitWeightedSampler,

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

    /// Idle noise after each two-qubit gate where noise will be applied stochastically based on
    /// `p2_idle`.
    ///
    /// This may be useful for memory sweeping.
    p2_idle: f64,

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
    /// Further details on how crosstalk is modeled depends on information from
    /// the device runtime, which are provided as `MeasCrosstalkGlobalPayload` instructions.
    p_meas_crosstalk_global: f64,

    /// Probability of crosstalk during measurement operations on local qubits
    ///
    /// See doc for `p_meas_crosstalk_global`. The intended distinction is that this
    /// parameter applies to only qubits that are close to the measured qubit.
    /// Further details on how crosstalk is modeled depends on information from
    /// the device runtime, which are provided as `MeasCrosstalkLocalPayload` instructions.
    p_meas_crosstalk_local: f64,

    /// Transition probabilities on the event of crosstalk error
    p_meas_crosstalk_model: CrosstalkWeightedSampler,

    // --- internally used variables --- //
    /// The maximum of `p_meas_0` and `p_meas_1`
    ///
    /// Used to determine the overall measurement error rate.
    p_meas_max: f64,

    /// Set of qubits that are currently in a leaked state
    ///
    /// Tracks which qubits have leaked out of the computational subspace and are
    /// therefore not affected by computational gates but might still affect measurements.
    leaked_qubits: BTreeSet<usize>,

    /// Random number generator for stochastic noise processes
    rng: NoiseRng<PecosRng>,

    /// Set of qubits that have been prepared at any point in the program.
    ///
    /// This is so that we know which qubits exists and we can apply crosstalk
    /// to them. Qubits that are measured / discarded are not removed from here, since
    /// PECOS does not assume measurements are destructive. This should not cause a
    /// problem, since inactive qubits suffering error have no effect on the state,
    /// and active qubits should always suffer errors under this naive crosstalk model.
    ///
    /// Using a `BTreeSet` because we will iterate over the qubits and we want determinism.
    prepared_qubits: BTreeSet<usize>,

    /// Track which qubits are being measured in the current batch and their gate types
    /// This is needed to properly handle leakage during measurements as well
    /// as crosstalk.
    measured_qubits: Vec<(usize, GateType)>,

    /// Stored outcome builder
    results_builder: ByteMessageBuilder,
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
        if self.results_builder.message_count() > 0 {
            return Err(PecosError::Processing(
                "Results builder not empty at start of processing".to_string(),
            ));
        }
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
        trace!("GeneralNoise::continue_processing");
        let next_operations = self.apply_noise_on_continue_processing(msg)?;
        if next_operations.is_empty()? {
            // No more quantum operations to process, return results
            // collected along the way and reset the results builder.
            let results = self.results_builder.build();
            self.results_builder.reset();
            Ok(EngineStage::Complete(results))
        } else {
            // if there are new quantum operations to process.
            Ok(EngineStage::NeedsProcessing(next_operations))
        }
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
    type Rng = PecosRng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.rng = NoiseRng::new(rng);
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
    /// use pecos_engines::noise::GeneralNoiseModel;
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
    ///
    /// # Errors
    ///
    /// Returns an error if noise application fails or the message cannot be processed.
    pub fn apply_noise_on_start(&mut self, input: &ByteMessage) -> Result<ByteMessage, String> {
        let mut builder = NoiseUtils::create_quantum_builder();
        let mut err = None;

        // Parse the input as quantum operations
        let gates = input
            .quantum_ops()
            .expect("Failed to parse input as quantum operations");

        for gate in gates {
            // Track which qubits are being measured for leakage handling
            if matches!(gate.gate_type, GateType::MZ | GateType::MeasureLeaked) {
                self.measured_qubits.extend(
                    gate.qubits
                        .iter()
                        .map(|q| (usize::from(*q), gate.gate_type)),
                );
            }

            // Skip noise application for noiseless gates
            if self.is_noiseless_gate(&gate.gate_type) {
                // Just add the gate as-is, without any noise
                // TODO: Still apply leakage rules
                builder.add_gate_command(&gate);
                trace!("Skipping noise for noiseless gate: {:?}", gate.gate_type);
                continue;
            }

            // For non-noiseless gates with qubits, we'll let the specific handlers
            // decide whether to add the original gate based on error models
            match gate.gate_type {
                GateType::Idle => {
                    self.apply_idle_faults(
                        &gate,
                        self.p_idle_linear_rate,
                        self.p_idle_quadratic_rate,
                        &mut builder,
                    );
                }
                GateType::PZ => {
                    for &q in &gate.qubits {
                        self.prepared_qubits.insert(usize::from(q));
                    }
                    self.apply_prep_faults(&gate, &mut builder);
                    self.apply_simple_crosstalk_faults(&gate, self.p_prep_crosstalk, &mut builder);
                }
                GateType::MZ | GateType::MeasureLeaked => {
                    // Measurement noise is handled in apply_noise_on_continue_processing
                    // We still need to add the original gate here
                    builder.add_gate_command(&gate);
                }
                GateType::MeasCrosstalkGlobalPayload => {
                    // Global crosstalk applies to all qubits that are *not* in the payload
                    let gate_qubits: Vec<usize> =
                        gate.qubits.iter().map(|q| usize::from(*q)).collect();
                    let potential_victims = self
                        .prepared_qubits
                        .iter()
                        .filter(|q| !gate_qubits.contains(q))
                        .copied()
                        .collect();

                    // Otherwise, it is the same channel for Global and Local
                    trace!("Applying global crosstalk...");
                    self.apply_crosstalk_faults_from_payload(
                        gate.gate_type,
                        potential_victims,
                        &mut builder,
                    );
                }
                GateType::MeasCrosstalkLocalPayload => {
                    let potential_victims = gate.qubits.iter().map(|q| usize::from(*q)).collect();

                    trace!("Applying local crosstalk...");
                    self.apply_crosstalk_faults_from_payload(
                        gate.gate_type,
                        potential_victims,
                        &mut builder,
                    );
                }
                GateType::I => {
                    let err_msg = format!(
                        "Identity is currently an unsupported gate type: {:?}",
                        gate.gate_type
                    );
                    err = Some(err_msg);
                }
                _ if gate.is_single_qubit() => {
                    self.apply_sq_faults(&gate, &mut builder);
                }
                _ if gate.is_two_qubit() => {
                    // For angle-dependent error rates (rotation gates like RZZ)
                    let p2 = if gate.angle_arity() >= 1 && !gate.angles.is_empty() {
                        let angle = gate.angles[0].to_radians();
                        self.p2_angle_error_rate(angle)
                    } else {
                        self.p2
                    };

                    self.apply_tq_faults(&gate, p2, &mut builder);
                }
                _ => {
                    // This should never happen since we've covered all cases above
                    let err_msg = format!("Unhandled gate type: {:?}", gate.gate_type);
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
    ///
    /// Note: Measurements do NOT unleak qubits. Only preparation operations unleak qubits.
    /// If a leaked qubit is measured, it remains leaked and will continue to measure as 1
    /// until a preparation operation is performed.
    ///
    /// Returns a `ByteMessage` destined for quantum simulation. Any results from measurements
    /// that are destined for the user are appended to `self.results_builder`.
    ///
    /// # Errors
    ///
    /// Returns an error if noise application fails or the message cannot be processed.
    pub fn apply_noise_on_continue_processing(
        &mut self,
        message: ByteMessage,
    ) -> Result<ByteMessage, PecosError> {
        // If there are no measurement results, return the message unchanged
        if !NoiseUtils::has_measurements(&message) {
            return Ok(message);
        }
        // Parse the measurements from the message
        let measurement_outcomes = message.outcomes()?;

        // Create a new message builder where the gates necessary for the
        // transitions will be introduced
        let mut ops_builder = ByteMessage::quantum_operations_builder();
        let mut outcomes = vec![];

        if measurement_outcomes.len() != self.measured_qubits.len() {
            return Err(PecosError::Processing(format!(
                "Mismatch in number of measurement outcomes ({}) and tracked measured qubits ({})",
                measurement_outcomes.len(),
                self.measured_qubits.len()
            )));
        }

        for (idx, outcome) in measurement_outcomes.into_iter().enumerate() {
            let (qubit, gate_type) = self.measured_qubits[idx];
            match gate_type {
                GateType::MeasCrosstalkGlobalPayload | GateType::MeasCrosstalkLocalPayload => {
                    // It is not a measurement destined for the user, but one we
                    // injected in order to model crosstalk. Use the measurement result
                    // to determine any transitions to apply.
                    let transition =
                        self.p_meas_crosstalk_model
                            .sample_gates(&mut self.rng, qubit, outcome);
                    if transition.has_leakage() {
                        if let Some(gate) = self.leak(qubit) {
                            ops_builder.add_gate_command(&gate);
                        }
                    } else if let Some(gate) = transition.gate {
                        ops_builder.add_gate_command(&gate);
                    }
                }
                GateType::MeasureLeaked => {
                    let mut val = outcome as usize;
                    if self.is_leaked(qubit) {
                        trace!("Qubit {qubit} is leaked, MeasureLeaked returns 2");
                        // For MeasureLeaked, return 2 for leaked qubits
                        val = 2;
                    } else if !self.noiseless_gates.contains(&GateType::MeasureLeaked) {
                        // For non-leaked qubits, apply measurement noise below
                        // Apply asymmetric measurement noise to each outcome
                        if val == 1 {
                            if self.rng.occurs(self.p_meas_1) {
                                trace!("Flipped measurement outcome 1->0");
                                val = 0;
                            }
                        } else if self.rng.occurs(self.p_meas_0) {
                            trace!("Flipped measurement outcome 0->1");
                            val = 1;
                        }
                    }
                    outcomes.push(val);
                }
                GateType::MZ => {
                    // Apply biased measurement noise to each outcome
                    // Check if we have leaked qubits that were measured
                    let mut val = outcome as usize;
                    if self.is_leaked(qubit) {
                        trace!("Qubit {qubit} is leaked, Measure returns 1");
                        // For regular Measure, force the measurement outcome to be 1
                        val = 1;
                    }
                    // NOTE: we still apply bit-flip noise to the outcome 1 of leaked
                    // qubits that measure as 1. This has been the approach since H1/H2.
                    if !self.noiseless_gates.contains(&GateType::MZ) {
                        if val == 1 {
                            if self.rng.occurs(self.p_meas_1) {
                                trace!("Flipped measurement outcome 1->0");
                                val = 0;
                            }
                        } else if self.rng.occurs(self.p_meas_0) {
                            trace!("Flipped measurement outcome 0->1");
                            val = 1;
                        }
                    }
                    outcomes.push(val);
                }
                GateType::PZ => {
                    // Just ignore the measurement.
                    // In the future we will want to do more advanced crosstalk
                    // on prep as well, but for now it uses apply_simple_crosstalk_faults
                    // where the measurement outcome is just meant to be thrown away.
                }
                _ => {
                    return Err(PecosError::Processing(format!(
                        "Unexpected gate type in measurement handling: {gate_type:?}"
                    )));
                }
            }
        }
        self.measured_qubits.clear();
        self.results_builder.add_outcomes(&outcomes);
        Ok(ops_builder.build())
    }

    pub fn apply_idle_faults(
        &mut self,
        gate: &Gate,
        linear_rate: f64,
        quadratic_rate: f64,
        builder: &mut ByteMessageBuilder,
    ) {
        if linear_rate > f64::EPSILON {
            let qubits_usize: Vec<usize> = gate.qubits.iter().map(|q| usize::from(*q)).collect();
            self.apply_idle_linear_stochastic_noise(
                linear_rate,
                gate.idle_duration(),
                &qubits_usize,
                builder,
            );
        }

        if quadratic_rate.abs() > f64::EPSILON {
            // TODO: add test
            let qubits_usize: Vec<usize> = gate.qubits.iter().map(|q| usize::from(*q)).collect();
            self.apply_idle_quadratic_dephasing(
                quadratic_rate,
                gate.idle_duration(),
                &qubits_usize,
                builder,
            );
        }
    }

    /// Assuming a general single-qubit stochastic noise for idling that depends on some rate and
    /// duration of idling (seconds).
    fn apply_idle_linear_stochastic_noise(
        &mut self,
        rate: f64,
        duration: f64,
        qubits: &[usize],
        builder: &mut ByteMessageBuilder,
    ) {
        let prob = rate * duration;
        for qubit in qubits {
            if !self.is_leaked(*qubit) && self.rng.occurs(prob) {
                let result = self.p_idle_linear_model.sample_gates(&mut self.rng, *qubit);

                if result.has_leakage() {
                    if let Some(gate) = self.leak(*qubit) {
                        builder.add_gate_command(&gate);
                    }
                } else if let Some(gate) = result.gate {
                    builder.add_gate_command(&gate);
                }
            }
        }
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
    /// * `angle` - The time duration over which idling occurs times the rate per time
    /// * `qubits` - The qubits that are potentially affected by the idling noise
    fn apply_idle_quadratic_dephasing(
        &mut self,
        rate: f64,
        duration: f64,
        qubits: &[usize],
        builder: &mut ByteMessageBuilder,
    ) {
        let mut angle = rate * duration;

        angle = if self.p_idle_coherent {
            angle
        } else {
            angle.sin().powi(2)
        };

        if angle.abs() > f64::EPSILON {
            let mut noisy_qubits = vec![];

            for qubit in qubits {
                if !self.is_leaked(*qubit) && (self.p_idle_coherent || self.rng.occurs(angle)) {
                    noisy_qubits.push(*qubit);
                }
            }
            if !noisy_qubits.is_empty() {
                if self.p_idle_coherent {
                    builder.rz(Angle64::from_radians(angle), &noisy_qubits);
                } else {
                    builder.z(&noisy_qubits);
                }
            }
        }
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
    pub fn apply_prep_faults(&mut self, gate: &Gate, builder: &mut ByteMessageBuilder) {
        // unleaking qubits - preparation resets leaked qubits to the zero state
        for &qubit in &gate.qubits {
            let qubit_usize = usize::from(qubit);
            if self.is_leaked(qubit_usize) {
                self.mark_as_unleaked(qubit_usize);
                trace!("Qubit {qubit} unleaked due to preparation");
            }
        }

        // Unlike SQ and TQ gates, state prep always occurs even if the qubit leaked
        builder.add_gate_command(gate);

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
                    if let Some(gate) = self.leak(usize::from(qubit)) {
                        builder.add_gate_command(&gate);
                    }
                    trace!("Qubit {qubit} leaked during preparation");
                } else {
                    builder.x(&[*qubit]);
                    trace!("Preparation error on qubit {qubit}");
                }
            }
        }
    }

    /// Apply simple crosstalk noise
    ///
    /// Naive crosstalk noise model:
    /// 1. All qubits in the trap but the ones in the `gate` are subject to crosstalk
    //     error. The `gate` should be either qubit measurement or preparation.
    //  2. *Each* qubit not in `gate` has the given `probability` to suffer an error.
    /// 3. Affected qubits are collapsed into the computational basis (Z measurement).
    ///
    /// In ion trap systems, this could represent scattered light during optical pumping
    /// affecting neighboring ions.
    pub fn apply_simple_crosstalk_faults(
        &mut self,
        gate: &Gate,
        probability: f64,
        builder: &mut ByteMessageBuilder,
    ) {
        let mut affected_qubits = Vec::new();
        let gate_qubits: Vec<usize> = gate.qubits.iter().map(|q| usize::from(*q)).collect();

        for q in self.prepared_qubits.clone() {
            if !gate_qubits.contains(&q) && self.rng.occurs(probability) {
                affected_qubits.push(q);
                trace!("Qubit {q} affected by crosstalk error");
            }
        }

        builder.mz(&affected_qubits);
        // We need to mark these measurements as being introduced by crosstalk rather
        // than the user's program so that we can discard the results in
        // apply_noise_on_continue_processing.
        self.measured_qubits
            .extend(affected_qubits.iter().map(|&q| (q, gate.gate_type)));
    }

    /// Apply crosstalk noise from runtime information given by Crosstalk*Payload instructions
    ///
    /// In ion trap systems, this could represent scattered light during optical pumping
    /// affecting neighboring ions.
    pub fn apply_crosstalk_faults_from_payload(
        &mut self,
        gate_type: GateType,
        qubits: Vec<usize>,
        builder: &mut ByteMessageBuilder,
    ) {
        let probability = match gate_type {
            GateType::MeasCrosstalkGlobalPayload => self.p_meas_crosstalk_global,
            GateType::MeasCrosstalkLocalPayload => self.p_meas_crosstalk_local,
            _ => unreachable!(),
        };

        let mut affected_qubits = Vec::new();

        for q in qubits {
            // TODO: We should include a seepage component to crosstalk in the future.
            if self.prepared_qubits.contains(&q)
                && !self.is_leaked(q) // If q is already leaked, we currently skip
                && self.rng.occurs(probability)
            {
                affected_qubits.push(q);
                trace!("Qubit {q} affected by crosstalk error");
            }
        }

        builder.mz(&affected_qubits);
        // We need to mark these measurements as being introduced by crosstalk rather
        // than the user's program so that we can discard the results in
        // apply_noise_on_continue_processing.
        self.measured_qubits
            .extend(affected_qubits.iter().map(|&q| (q, gate_type)));
        // NOTE: Crosstalk transitions are carried out by apply_noise_on_continue_processing
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
    pub fn apply_sq_faults(&mut self, gate: &Gate, builder: &mut ByteMessageBuilder) {
        let mut noise = Vec::new();
        let mut removed_gates = false;
        let mut original_gate_qubits: Vec<usize> = Vec::new();

        for &qubit in &gate.qubits {
            // Track whether to add the original gate
            let mut add_original_gate = true;
            let has_leakage = self.is_leaked(usize::from(qubit));

            if has_leakage {
                add_original_gate = false;
            }

            if self.rng.occurs(self.p1) {
                // Spontaneous emission
                if self.rng.occurs(self.p1_emission_ratio) {
                    // If qubit has leaked and spontaneous emission has occurred... seep the qubit
                    if has_leakage {
                        if let Some(gates) = self.seep(usize::from(qubit), self.p1_seepage_prob) {
                            noise.extend(gates);
                        }
                    } else {
                        add_original_gate = false;

                        let result = self
                            .p1_emission_model
                            .sample_gates(&mut self.rng, usize::from(qubit));

                        if result.has_leakage() {
                            // Handle leakage
                            if let Some(gate) = self.leak(usize::from(qubit)) {
                                noise.push(gate);
                            }
                        } else if let Some(gate) = result.gate {
                            // Handle Pauli gate
                            noise.push(gate);
                            trace!("Applied Pauli error to qubit {qubit}");
                        }
                    }
                } else if !has_leakage {
                    // Pauli noise
                    // TODO: Check if there is any assurance that the model is only Pauli noise
                    let result = self
                        .p1_pauli_model
                        .sample_gates(&mut self.rng, usize::from(qubit));
                    if let Some(gate) = result.gate {
                        noise.push(gate);
                        trace!("Applied Pauli error to qubit {qubit}");
                    }
                }
            }

            // Add the original gate only if there were no leakage errors
            if add_original_gate {
                original_gate_qubits.push(usize::from(qubit));
            } else {
                removed_gates = true;
            }
        }

        if removed_gates {
            // There are some gates left to add
            if !original_gate_qubits.is_empty() {
                let qubits_qubit_id: Vec<QubitId> =
                    original_gate_qubits.into_iter().map(QubitId).collect();
                let new_gate = Gate::new(
                    gate.gate_type,
                    gate.angles.clone(),
                    gate.params.clone(),
                    qubits_qubit_id,
                );
                builder.add_gate_command(&new_gate);
            }
        } else {
            builder.add_gate_command(gate);
        }

        if !noise.is_empty() {
            builder.add_gate_commands(&noise);
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
    pub fn apply_tq_faults(&mut self, gate: &Gate, p: f64, builder: &mut ByteMessageBuilder) {
        let mut noise = Vec::new();
        let mut removed_gates = false;
        let mut original_gate_qubits: Vec<usize> = Vec::new();

        for qubits in gate.qubits.chunks_exact(2) {
            let mut add_original_gate = true;

            // Check if the gate is acting on a leaked qubit in a way to
            let has_leakage = !self.leaked_qubits.is_empty()
                && gate
                    .qubits
                    .iter()
                    .any(|&qubit| self.is_leaked(usize::from(qubit)));

            if has_leakage {
                add_original_gate = false;
            }

            if self.rng.occurs(p) {
                if self.rng.occurs(self.p2_emission_ratio) {
                    if has_leakage {
                        // potentially seep qubits
                        for qubit in &gate.qubits {
                            if self.is_leaked(usize::from(*qubit))
                                && let Some(gates) =
                                    self.seep(usize::from(*qubit), self.p2_seepage_prob)
                            {
                                noise.extend(gates);
                            }
                        }
                    } else {
                        // Spontaneous emission noise
                        add_original_gate = false;

                        let result = self.p2_emission_model.sample_gates(
                            &mut self.rng,
                            usize::from(qubits[0]),
                            usize::from(qubits[1]),
                        );

                        if result.has_leakage() {
                            for (qubit, leaked) in qubits.iter().zip(result.has_leakages().iter()) {
                                if *leaked && let Some(gate) = self.leak(usize::from(*qubit)) {
                                    noise.push(gate);
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
                    let result = self.p2_pauli_model.sample_gates(
                        &mut self.rng,
                        usize::from(qubits[0]),
                        usize::from(qubits[1]),
                    );
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
                original_gate_qubits.extend(qubits.iter().map(|q| usize::from(*q)));
            } else {
                removed_gates = true;
            }
        }

        if removed_gates {
            // There are some gates left to add
            if !original_gate_qubits.is_empty() {
                let qubits_qubit_id: Vec<QubitId> =
                    original_gate_qubits.iter().map(|&q| QubitId(q)).collect();
                let new_gate = Gate::new(
                    gate.gate_type,
                    gate.angles.clone(),
                    gate.params.clone(),
                    qubits_qubit_id,
                );
                builder.add_gate_command(&new_gate);
            }
        } else {
            builder.add_gate_command(gate);
        }

        builder.add_gate_commands(&noise);

        if self.p2_idle > f64::EPSILON {
            self.apply_idle_linear_stochastic_noise(
                self.p2_idle,
                1.0,
                &original_gate_qubits,
                builder,
            );
        }
    }

    /// Leak a qubit (or replace it with completely depolarizing noise)
    ///
    /// When a qubit leaks, it moves outside the computational subspace and can no longer be
    /// affected by quantum gates, but may still be re-prepared and measured.
    /// Here we have the chance to replace the leakage event with completely depolarizing noise...
    /// `self.leakage_scale` acts like the probability to apply leakage instead of completely
    /// depolarizing noise.
    fn leak(&mut self, qubit: usize) -> Option<Gate> {
        if self.leakage_scale >= 1.0 || self.rng.occurs(self.leakage_scale) {
            // Mark qubit as leaked
            trace!("Marking qubit {qubit} as leaked");
            self.mark_as_leaked(qubit);
            Some(Gate::pz(&[qubit]))
        } else {
            // Apply completely depolarizing noise instead of leakage
            trace!("Replaced leakage with Pauli error on qubit {qubit}");
            self.rng.random_pauli_or_none(qubit)
        }
    }

    /// Mark a qubit as leaked
    ///
    /// This method explicitly sets a qubit to the leaked state, simulating a qubit that has
    /// transitioned outside the computational basis (e.g., to a higher energy level in ion traps).
    ///
    /// # Behavior of Leaked Qubits
    ///
    /// - Regular `Measure` operations on leaked qubits always return 1
    /// - `MeasureLeaked` operations on leaked qubits return 2
    /// - Leaked qubits remain leaked until a `Prep` operation is applied
    /// - Gates applied to leaked qubits have no effect on the quantum state
    ///
    /// # Use Cases
    ///
    /// - Testing algorithm robustness against leakage errors
    /// - Setting up specific initial conditions for simulations
    /// - Research into leakage-aware quantum algorithms
    /// - Debugging circuits with leakage errors
    ///
    /// # Example
    ///
    /// ```rust
    /// use pecos_engines::noise::GeneralNoiseModel;
    ///
    /// // Create a noise model for testing leakage
    /// let mut noise_model = GeneralNoiseModel::default();
    ///
    /// // Mark qubit 0 as leaked - useful for testing leakage-aware algorithms
    /// noise_model.mark_as_leaked(0);
    ///
    /// // Mark multiple qubits as leaked for batch testing
    /// noise_model.mark_as_leaked(1);
    /// noise_model.mark_as_leaked(3);
    ///
    /// // The noise model now tracks these qubits as leaked
    /// // This affects how noise operations are applied during simulation
    /// ```
    pub fn mark_as_leaked(&mut self, qubit: usize) {
        // TODO: see if some of the mark_as_leaked needs to move to self.leak()
        trace!("Marking qubit {qubit} as leaked");
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

    fn unleak(&mut self, qubit: usize) -> Option<Gate> {
        trace!("Replaced leakage with Pauli error on qubit {qubit}");
        if self.leakage_scale == 0.0 {
            // No leakage is being applied in the system
            None
        } else {
            trace!("Marking qubit {qubit} as unleaked");
            self.mark_as_unleaked(qubit);
            Option::from(Gate::pz(&[qubit]))
        }
    }

    fn unleak_random_bit(&mut self, qubit: usize) -> Vec<Gate> {
        let mut noise = vec![];

        if let Some(gate) = self.unleak(qubit) {
            noise.push(gate);
        }

        if let Some(gate) = self.rng.random_pauli_or_none(qubit) {
            noise.push(gate);
        }

        noise
    }

    fn seep(&mut self, qubit: usize, seepage_prob: f64) -> Option<Vec<Gate>> {
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
        // Clear measured qubits
        self.measured_qubits.clear();
        // RNG state is intentionally not reset to maintain natural randomness
    }

    /// Calculate the two-qubit gate error rate based on the rotation angle
    ///
    /// with additional support for asymmetric scaling and power-law scaling
    #[must_use]
    pub fn p2_angle_error_rate(&self, angle: f64) -> f64 {
        // Normalize angle by π - convert to a value in [0, 1] range
        let theta = angle.abs() / std::f64::consts::PI;

        // Apply power scaling to the normalized theta
        let theta_power = theta.powf(self.p2_angle_power);

        // Determine base rate based on angle sign
        let base_rate = if angle < 0.0 {
            // Negative angle - use a and b coefficients
            self.p2_angle_a * theta_power + self.p2_angle_b
        } else if angle > 0.0 {
            // Positive angle - use c and d coefficients
            self.p2_angle_c * theta_power + self.p2_angle_d
        } else {
            // Angle is exactly zero - use average of b and d
            (self.p2_angle_b + self.p2_angle_d) * 0.5
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
    pub fn reset_with_seed(&mut self, seed: u64) {
        // First reset the noise model
        self.reset_noise_model();
        // Then set the seed
        self.set_seed(seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gate;
    use crate::byte_message::ByteMessageBuilder;
    use crate::byte_message::GateType;
    use pecos_core::Angle64;

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

    /// Helper function to invoke a measurement request from the user to the noise
    /// model, and respond back with measurement values from the user. The result
    /// is the engine stage after the measurement results have been processed.
    fn request_measurements_and_give_values(
        noise: &mut GeneralNoiseModel,
        qubits: &[usize],
        values: &[usize],
    ) -> EngineStage<ByteMessage, ByteMessage> {
        use crate::byte_message::ByteMessageBuilder;
        // Create a measurement operation on the given qubits
        let mut request_builder = ByteMessageBuilder::new();
        let _ = request_builder.for_quantum_operations();
        for &qubit in qubits {
            request_builder.add_gate_command(&Gate {
                gate_type: GateType::MZ,
                angles: vec![].into(),
                qubits: vec![QubitId(qubit)].into(),
                params: vec![].into(),
            });
        }
        let measurement_request = request_builder.build();

        // Send the measurement requests to the noise model
        let Ok(EngineStage::NeedsProcessing(_)) = noise.start(measurement_request) else {
            panic!("Expected NeedsProcessing stage after measurement with noise");
        };

        // Now create the measurement results
        let mut response_builder = ByteMessageBuilder::new();
        let _ = response_builder.for_outcomes();
        response_builder.add_outcomes(values);
        let measurement_response = response_builder.build();

        // Process the measurement results through the noise model
        noise.continue_processing(measurement_response).unwrap()
    }
    #[test]
    fn test_biased_measurement() {
        // Create a noise model with 100% flip probabilities,
        // and a noise model with 0% flip probabilities. Have them
        // both measure two qubits, one reporting state 0 and one in state 1.
        // Ensure that the biased model flips both results, while the unbiased
        // model leaves them unchanged.

        // The clean model
        let mut never_flip = GeneralNoiseModel::new(0.0, 0.0, 0.0, 0.0, 0.0);
        let never_flip_state =
            request_measurements_and_give_values(&mut never_flip, &[0, 1], &[0, 1]);
        let EngineStage::Complete(clean_results) = never_flip_state else {
            panic!("Expected results from the unbiased noise model");
        };
        let clean_outcomes = clean_results.outcomes().unwrap();
        assert_eq!(
            clean_outcomes.len(),
            2,
            "Should have two measurement results from the unbaised model"
        );
        assert_eq!(clean_outcomes[0], 0, "0 shouldn't be flipped to 1");
        assert_eq!(clean_outcomes[1], 1, "1 shouldn't be flipped to 0");

        // The noisy model
        let mut always_flip = GeneralNoiseModel::new(0.0, 1.0, 1.0, 0.0, 0.0);
        let always_flip_state =
            request_measurements_and_give_values(&mut always_flip, &[0, 1], &[0, 1]);
        let EngineStage::Complete(noisy_results) = always_flip_state else {
            panic!("Expected results from the biased noise model");
        };
        let noisy_outcomes = noisy_results.outcomes().unwrap();
        assert_eq!(
            noisy_outcomes.len(),
            2,
            "Should have two measurement results from the unbaised model"
        );
        assert_eq!(noisy_outcomes[0], 1, "0 should be flipped to 1");
        assert_eq!(noisy_outcomes[1], 0, "1 should be flipped to 0");
    }

    #[test]
    fn test_prep_leak_ratio() {
        use crate::Gate;
        use crate::byte_message::{ByteMessageBuilder, GateType};

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
        let gate = Gate {
            gate_type: GateType::PZ,
            angles: vec![].into(),
            qubits: vec![QubitId(0)].into(),
            params: vec![].into(),
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
        // Create a noise model with no spontaneous errors
        let mut model = GeneralNoiseModel::builder()
            .with_prep_probability(0.0)
            .with_meas_0_probability(0.0)
            .with_meas_1_probability(0.0)
            .with_p1_probability(0.0)
            .with_p2_probability(0.0)
            .build();

        // Manually mark qubit 0 as leaked
        model.mark_as_leaked(0);

        // First, tell the noise model that we are measuring qubit 0, and give it value 0
        let state = request_measurements_and_give_values(&mut model, &[0], &[0]);
        let EngineStage::Complete(biased_message) = state else {
            panic!("Expected Complete stage after measurement with noise");
        };

        // Get the measurement results
        let outcomes = biased_message.outcomes().unwrap();
        assert_eq!(outcomes.len(), 1, "Should have one measurement result");

        // Verify that the leaked qubit is reported as measured as 1
        assert_eq!(outcomes[0], 1, "Leaked qubit should always measure as 1");

        // Verify that the qubit is still leaked after measurement
        // Measurements do not unleak qubits - only prep operations do
        assert!(
            model.is_leaked(0),
            "Qubit should remain leaked after measurement"
        );
    }

    #[test]
    fn test_repeated_measurement_of_leaked_qubit() {
        // Create a noise model with no spontaneous errors
        let mut model = GeneralNoiseModel::builder()
            .with_prep_probability(0.0)
            .with_meas_0_probability(0.0)
            .with_meas_1_probability(0.0)
            .with_p1_probability(0.0)
            .with_p2_probability(0.0)
            .build();

        // Manually mark qubit 0 as leaked
        model.mark_as_leaked(0);

        // Measure 0 thrice and provide measurement result 0 each time
        let state = request_measurements_and_give_values(&mut model, &[0, 0, 0], &[0, 0, 0]);
        let EngineStage::Complete(biased_message) = state else {
            panic!("Expected Complete stage after measurement with noise");
        };
        // Get the measurement results
        let outcomes = biased_message.outcomes().unwrap();
        let results: Vec<(usize, u32)> = outcomes.into_iter().enumerate().collect();

        // Verify that all three measurements of the leaked qubit report as 1
        assert_eq!(results.len(), 3, "Should have three measurement results");
        assert_eq!(
            results[0].1, 1,
            "First measurement of leaked qubit should be 1"
        );
        assert_eq!(
            results[1].1, 1,
            "Second measurement of leaked qubit should be 1"
        );
        assert_eq!(
            results[2].1, 1,
            "Third measurement of leaked qubit should be 1"
        );

        // Verify that the qubit is still leaked after all measurements
        assert!(
            model.is_leaked(0),
            "Qubit should remain leaked after repeated measurements"
        );
    }

    #[test]
    fn test_prep_unleaks_after_measurement() {
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
        assert!(noise.is_leaked(0), "Qubit should start as leaked");

        // Process a measurement gate
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.mz(&[0]);
        let measurement_command = builder.build();
        let _noisy_command = noise.start(measurement_command).unwrap();

        // Process measurement results
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();
        builder.add_outcomes(&[0]);
        let state = noise.continue_processing(builder.build()).unwrap();
        let EngineStage::Complete(biased_message) = state else {
            panic!("Expected Complete stage after measurement with noise");
        };

        // Verify the leaked qubit measured as 1 but remains leaked
        let outcomes = biased_message.outcomes().unwrap();
        let results: Vec<(usize, u32)> = outcomes.into_iter().enumerate().collect();
        assert_eq!(results.len(), 1, "Should have one measurement result");
        assert_eq!(results[0].1, 1, "Leaked qubit should measure as 1");
        assert!(
            noise.is_leaked(0),
            "Qubit should remain leaked after measurement"
        );

        // Now apply a prep operation
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        let prep_gate = Gate {
            gate_type: GateType::PZ,
            angles: vec![].into(),
            qubits: vec![QubitId(0)].into(),
            params: vec![].into(),
        };
        noise.apply_prep_faults(&prep_gate, &mut builder);

        // Verify that the qubit is now unleaked
        assert!(
            !noise.is_leaked(0),
            "Qubit should be unleaked after prep operation"
        );
    }

    #[test]
    fn test_measurement_order_preservation() {
        use crate::byte_message::ByteMessageBuilder;

        // Create a noise model with biased measurement probabilities
        let mut model = GeneralNoiseModel::builder()
            .with_meas_0_probability(0.3) // 30% chance of flipping 0 to 1
            .with_meas_1_probability(0.2) // 20% chance of flipping 1 to 0
            .with_seed(42) // Use fixed seed for deterministic test
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Create measurement gates for different qubits in specific order
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.mz(&[2]); // First: measure qubit 2
        builder.mz(&[0]); // Second: measure qubit 0
        builder.mz(&[3]); // Third: measure qubit 3
        builder.mz(&[1]); // Fourth: measure qubit 1
        builder.mz(&[2]); // Fifth: measure qubit 2 again
        let measurement_command = builder.build();

        // Process the measurement gates through the noise model
        let _noisy_command = noise.start(measurement_command).unwrap();

        // Create measurement results in the same order
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();
        builder.add_outcomes(&[1, 0, 1, 0, 1]); // Results in order

        // Apply measurement noise
        let state = noise.continue_processing(builder.build()).unwrap();
        let EngineStage::Complete(noisy_results) = state else {
            panic!("Expected Complete stage after measurement with noise");
        };

        // Parse the noisy results
        let results = noisy_results.outcomes().unwrap();

        // Verify we have the correct number of results
        assert_eq!(results.len(), 5, "Should have 5 measurement results");

        // The order should be preserved even with noise
        // Results might be flipped due to noise, but the order should remain:
        // results[0] corresponds to qubit 2 (first measurement)
        // results[1] corresponds to qubit 0 (second measurement)
        // results[2] corresponds to qubit 3 (third measurement)
        // results[3] corresponds to qubit 1 (fourth measurement)
        // results[4] corresponds to qubit 2 (fifth measurement)

        // Print results for debugging
        println!("Original: [1, 0, 1, 0, 1]");
        println!("Noisy:    {results:?}");

        // Check that the noise model tracked the correct qubits
        // Note: measured_qubits is cleared after processing, so we can't check it here
        // But we can verify the results are in the expected range
        for (i, &result) in results.iter().enumerate() {
            assert!(
                result == 0 || result == 1,
                "Result {i} should be 0 or 1, got {result}"
            );
        }
    }

    #[test]
    fn test_measurement_order_with_leakage() {
        use crate::byte_message::ByteMessageBuilder;

        // Create a noise model with no measurement errors
        let mut model = GeneralNoiseModel::builder()
            .with_meas_0_probability(0.0)
            .with_meas_1_probability(0.0)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Mark specific qubits as leaked
        noise.mark_as_leaked(1);
        noise.mark_as_leaked(3);

        // Create measurement gates in specific order
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.mz(&[0]); // Non-leaked
        builder.mz(&[1]); // Leaked
        builder.mz(&[2]); // Non-leaked
        builder.mz(&[3]); // Leaked
        builder.mz(&[1]); // Leaked (repeated)
        let measurement_command = builder.build();

        // Process the measurement gates
        let _noisy_command = noise.start(measurement_command).unwrap();

        // Create measurement results (all zeros)
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();
        builder.add_outcomes(&[0, 0, 0, 0, 0]);

        // Apply noise (should force leaked qubits to 1)
        let state = noise.continue_processing(builder.build()).unwrap();
        let EngineStage::Complete(noisy_results) = state else {
            panic!("Expected Complete stage after measurement with noise");
        };

        // Parse results
        let results = noisy_results.outcomes().unwrap();

        // Verify order and leakage effects
        assert_eq!(results.len(), 5, "Should have 5 measurement results");
        assert_eq!(results[0], 0, "Qubit 0 (non-leaked) should remain 0");
        assert_eq!(results[1], 1, "Qubit 1 (leaked) should be forced to 1");
        assert_eq!(results[2], 0, "Qubit 2 (non-leaked) should remain 0");
        assert_eq!(results[3], 1, "Qubit 3 (leaked) should be forced to 1");
        assert_eq!(
            results[4], 1,
            "Qubit 1 (leaked, repeated) should be forced to 1"
        );
    }

    #[test]
    fn test_biased_measurement_statistics() {
        // Test with many measurements to see clear statistical pattern
        const NUM_MEASUREMENTS: usize = 1000;

        // Create a noise model with strong asymmetric bias
        // 80% chance of flipping 0->1, only 10% chance of flipping 1->0
        let mut model = GeneralNoiseModel::builder()
            .with_meas_0_probability(0.8) // Strong bias: 0 -> 1
            .with_meas_1_probability(0.1) // Weak bias: 1 -> 0
            .with_seed(12345) // Fixed seed for reproducibility
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // First test: all zeros
        let mut zeros_flipped = 0;
        for i in 0..NUM_MEASUREMENTS {
            // Need to process measurement gate first
            let state = request_measurements_and_give_values(noise, &[0], &[0]);
            let EngineStage::Complete(biased_result) = state else {
                panic!("Expected Complete stage after measurement with noise");
            };
            let results = biased_result.outcomes().unwrap();
            assert_eq!(
                results.len(),
                1,
                "Expected one measurement result per iteration"
            );

            if results[0] == 1 {
                zeros_flipped += 1;
            }

            // Reset for next measurement
            noise.reset_noise_model();

            // Reset seed periodically to get different random values
            if i % 100 == 99 {
                noise.set_seed(12345 + (i / 100) as u64);
            }
        }

        // Expect approximately 80% of zeros to flip to ones
        let zero_flip_rate = f64::from(zeros_flipped)
            / f64::from(u32::try_from(NUM_MEASUREMENTS).unwrap_or(u32::MAX));
        println!(
            "Zero flip rate: {:.1}% (expected ~80%)",
            zero_flip_rate * 100.0
        );
        assert!(
            (zero_flip_rate - 0.8).abs() < 0.05,
            "Zero flip rate {zero_flip_rate:.3} should be close to 0.8"
        );

        // Second test: all ones
        let mut ones_flipped = 0;
        noise.set_seed(54321); // Different seed for variety

        for i in 0..NUM_MEASUREMENTS {
            // Process measurement gate
            let state = request_measurements_and_give_values(noise, &[0], &[1]);
            let EngineStage::Complete(biased_result) = state else {
                panic!("Expected Complete stage after measurement with noise");
            };
            let results = biased_result.outcomes().unwrap();

            if results[0] == 0 {
                ones_flipped += 1;
            }

            // Reset for next measurement
            noise.reset_noise_model();

            // Reset seed periodically
            if i % 100 == 99 {
                noise.set_seed(54321 + (i / 100) as u64);
            }
        }

        // Expect approximately 10% of ones to flip to zeros
        let one_flip_rate = f64::from(ones_flipped)
            / f64::from(u32::try_from(NUM_MEASUREMENTS).unwrap_or(u32::MAX));
        println!(
            "One flip rate: {:.1}% (expected ~10%)",
            one_flip_rate * 100.0
        );
        assert!(
            (one_flip_rate - 0.1).abs() < 0.05,
            "One flip rate {one_flip_rate:.3} should be close to 0.1"
        );
    }

    #[test]
    fn test_extreme_measurement_bias() {
        use crate::byte_message::ByteMessageBuilder;

        // Test with extreme biases to make the effect very clear
        // Case 1: Always flip 0->1, never flip 1->0
        let mut model = GeneralNoiseModel::builder()
            .with_meas_0_probability(1.0) // Always flip 0->1
            .with_meas_1_probability(0.0) // Never flip 1->0
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Test batch of mixed measurements
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        for i in 0..10 {
            builder.mz(&[i]);
        }
        let _cmd = noise.start(builder.build()).unwrap();

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();
        // Original pattern: 0,1,0,1,0,1,0,1,0,1
        builder.add_outcomes(&[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);

        let state = noise.continue_processing(builder.build()).unwrap();
        let EngineStage::Complete(biased_result) = state else {
            panic!("Expected Complete stage after measurement with noise");
        };
        let results = biased_result.outcomes().unwrap();

        // Expected pattern after noise: 1,1,1,1,1,1,1,1,1,1 (all zeros flipped)
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(
                result, 1,
                "Position {i}: With 100% 0->1 flip and 0% 1->0 flip, all should be 1"
            );
        }

        // Case 2: Never flip 0->1, always flip 1->0
        let mut model = GeneralNoiseModel::builder()
            .with_meas_0_probability(0.0) // Never flip 0->1
            .with_meas_1_probability(1.0) // Always flip 1->0
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Reset and test again
        noise.reset_noise_model();

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        for i in 0..10 {
            builder.mz(&[i]);
        }
        let _cmd = noise.start(builder.build()).unwrap();

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();
        // Same original pattern: 0,1,0,1,0,1,0,1,0,1
        builder.add_outcomes(&[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);

        let state = noise.continue_processing(builder.build()).unwrap();
        let EngineStage::Complete(biased_result) = state else {
            panic!("Expected Complete stage after measurement with noise");
        };
        let results = biased_result.outcomes().unwrap();

        // Expected pattern after noise: 0,0,0,0,0,0,0,0,0,0 (all ones flipped)
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(
                result, 0,
                "Position {i}: With 0% 0->1 flip and 100% 1->0 flip, all should be 0"
            );
        }
    }

    #[test]
    fn test_measure_leaked_without_leakage() {
        use crate::byte_message::ByteMessageBuilder;

        // Create a noise model with no errors (deterministic)
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

        // No qubits are leaked
        assert!(!noise.is_leaked(0));
        assert!(!noise.is_leaked(1));

        // Create measurement gates with both Measure and MeasureLeaked
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.mz(&[0]); // Regular measure
        builder.measure_leakages(&[1]); // MeasureLeaked

        let measurement_command = builder.build();
        let _noisy_command = noise.start(measurement_command.clone()).unwrap();

        // Create measurement results (both qubits in |0⟩ state)
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();
        builder.add_outcomes(&[0, 0]);

        let state = noise.continue_processing(builder.build()).unwrap();
        let EngineStage::Complete(results_message) = state else {
            panic!("Expected Complete stage after measurement with noise");
        };
        let results = results_message.outcomes().unwrap();
        assert_eq!(results.len(), 2, "Should have two measurement results");
        assert_eq!(results[0], 0, "Regular Measure of |0⟩ should return 0");
        assert_eq!(
            results[1], 0,
            "MeasureLeaked of |0⟩ should return 0 (not leaked)"
        );

        // Test with |1⟩ state
        let _noisy_command = noise.start(measurement_command.clone()).unwrap();
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();
        builder.add_outcomes(&[1, 1]);

        let state = noise.continue_processing(builder.build()).unwrap();
        let EngineStage::Complete(results_message) = state else {
            panic!("Expected Complete stage after measurement with noise");
        };

        let results = results_message.outcomes().unwrap();
        assert_eq!(results[0], 1, "Regular Measure of |1⟩ should return 1");
        assert_eq!(
            results[1], 1,
            "MeasureLeaked of |1⟩ should return 1 (not leaked)"
        );
    }

    #[test]
    fn test_measure_leaked_with_leakage() {
        use crate::byte_message::ByteMessageBuilder;

        // Create a noise model with no measurement errors (deterministic)
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

        // Manually mark qubits 0 and 1 as leaked
        noise.mark_as_leaked(0);
        noise.mark_as_leaked(1);

        // Create measurement gates with both Measure and MeasureLeaked
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.mz(&[0]); // Regular measure on leaked qubit
        builder.measure_leakages(&[1]); // MeasureLeaked on leaked qubit

        let measurement_command = builder.build();
        let _noisy_command = noise.start(measurement_command).unwrap();

        // Create measurement results (simulator returns 0, but noise model will override)
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();
        builder.add_outcomes(&[0, 0]);

        let state = noise.continue_processing(builder.build()).unwrap();
        let EngineStage::Complete(results_message) = state else {
            panic!("Expected Complete stage after measurement with noise");
        };

        let results = results_message.outcomes().unwrap();
        assert_eq!(results.len(), 2, "Should have two measurement results");
        assert_eq!(
            results[0], 1,
            "Regular Measure of leaked qubit should return 1"
        );
        assert_eq!(
            results[1], 2,
            "MeasureLeaked of leaked qubit should return 2"
        );
    }

    #[test]
    fn test_measure_leaked_mixed_scenario() {
        use crate::byte_message::ByteMessageBuilder;

        // Create a noise model with no measurement errors (deterministic)
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

        // Mark only even qubits as leaked
        noise.mark_as_leaked(0);
        noise.mark_as_leaked(2);

        // Create mixed measurement gates
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.mz(&[0]); // Regular measure on leaked qubit 0
        builder.measure_leakages(&[1]); // MeasureLeaked on non-leaked qubit 1
        builder.measure_leakages(&[2]); // MeasureLeaked on leaked qubit 2
        builder.mz(&[3]); // Regular measure on non-leaked qubit 3

        let measurement_command = builder.build();
        let _noisy_command = noise.start(measurement_command).unwrap();

        // Create measurement results (mix of 0s and 1s from simulator)
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();
        builder.add_outcomes(&[0, 1, 0, 1]); // Simulator results before noise

        let state = noise.continue_processing(builder.build()).unwrap();
        let EngineStage::Complete(results_message) = state else {
            panic!("Expected Complete stage after measurement with noise");
        };

        let results = results_message.outcomes().unwrap();
        assert_eq!(results.len(), 4, "Should have 4 measurement results");
        assert_eq!(results[0], 1, "Measure on leaked qubit 0 should return 1");
        assert_eq!(
            results[1], 1,
            "MeasureLeaked on non-leaked qubit 1 should preserve simulator result 1"
        );
        assert_eq!(
            results[2], 2,
            "MeasureLeaked on leaked qubit 2 should return 2"
        );
        assert_eq!(
            results[3], 1,
            "Measure on non-leaked qubit 3 should preserve simulator result 1"
        );
    }

    #[test]
    fn test_measurement_bias_with_leakage() {
        use crate::byte_message::ByteMessageBuilder;

        // Test that leaked qubits are forced to 1, then bias is applied
        let mut model = GeneralNoiseModel::builder()
            .with_meas_0_probability(0.0) // No 0->1 flips
            .with_meas_1_probability(0.5) // 50% chance to flip 1->0
            .with_seed(42)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Mark some qubits as leaked
        noise.mark_as_leaked(0);
        noise.mark_as_leaked(2);

        // Process measurements
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.mz(&[0, 1, 2, 3]); // 0 and 2 are leaked
        let _cmd = noise.start(builder.build()).unwrap();

        // All original results are 0
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();
        builder.add_outcomes(&[0, 0, 0, 0]);

        // Run many times to see statistics
        let mut leaked_flipped_to_zero = 0;
        let runs = 100;

        for i in 0..runs {
            // Reset noise model state but keep leaked qubits
            noise.measured_qubits.clear();
            noise.set_seed(42 + i);

            // Re-process measurement gates each time
            let mut gate_builder = ByteMessageBuilder::new();
            let _ = gate_builder.for_quantum_operations();
            gate_builder.mz(&[0, 1, 2, 3]);
            let _cmd = noise.start(gate_builder.build()).unwrap();

            let state = noise.continue_processing(builder.build()).unwrap();
            let EngineStage::Complete(biased_result) = state else {
                panic!("Expected Complete stage after measurement with noise");
            };
            let results = biased_result.outcomes().unwrap();
            assert_eq!(results.len(), 4, "Expected 4 measurement results");

            // Qubits 0 and 2 were leaked, so forced to 1, then 50% chance to flip to 0
            if results[0] == 0 {
                leaked_flipped_to_zero += 1;
            }

            // Regular qubits 1 and 3 should remain 0 (no 0->1 bias)
            assert_eq!(
                results[1], 0,
                "Non-leaked qubit with 0 result and no 0->1 bias should stay 0"
            );
            assert_eq!(
                results[3], 0,
                "Non-leaked qubit with 0 result and no 0->1 bias should stay 0"
            );
        }

        // Leaked qubits should be ~50/50 due to 50% 1->0 flip probability
        let flip_rate =
            f64::from(leaked_flipped_to_zero) / f64::from(u32::try_from(runs).unwrap_or(u32::MAX));
        println!(
            "Leaked qubit 1->0 flip rate: {:.1}% (expected ~50%)",
            flip_rate * 100.0
        );
        assert!(
            (flip_rate - 0.5).abs() < 0.15,
            "Leaked qubit flip rate {flip_rate:.3} should be close to 0.5"
        );
    }

    #[test]
    fn test_prep_crosstalk() {
        use crate::byte_message::ByteMessageBuilder;

        let mut model = GeneralNoiseModel::builder()
            .with_p_prep_crosstalk(1.0)
            .with_seed(42)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        // Prepare a bunch of |0> states
        builder.pz(&[0, 1, 2, 3, 4]);
        // Apply mid-circuit measurement and reset
        builder.mz(&[2]);
        builder.pz(&[2]);
        let _cmd = noise.start(builder.build()).unwrap();

        assert_eq!(
            noise.measured_qubits.len(),
            5,
            "There should be 5 measured qubits: one from MCMR and the others from
            crosstalk got: {:?}",
            noise.measured_qubits
        );

        let (q, gate_type) = noise.measured_qubits[0];
        assert_eq!(q, 2, "The first measurement should be the MCMR on qubit 2");
        assert!(
            gate_type == GateType::MZ,
            "The first measurement should come from MCMR"
        );

        for (_, gate_type) in &noise.measured_qubits[1..] {
            assert!(
                *gate_type == GateType::PZ,
                "The other measurements should come from crosstalk"
            );
        }

        // All results are 0
        let mut outcome_builder = ByteMessageBuilder::new();
        let _ = outcome_builder.for_outcomes();
        outcome_builder.add_outcomes(&[0, 0, 0, 0, 0]);

        let mcmr = noise.continue_processing(outcome_builder.build()).unwrap();

        let EngineStage::Complete(mcmr) = mcmr else {
            panic!("Expected Complete stage after processing outcomes");
        };
        let results = mcmr.outcomes().unwrap();

        assert_eq!(
            noise.measured_qubits.len(),
            0,
            "The list of measured_qubits should have been cleared."
        );
        assert_eq!(
            results.len(),
            1,
            "There should only be one outcome: that of the mid-circ measurement"
        );
    }

    #[test]
    fn test_meas_crosstalk() {
        use crate::byte_message::ByteMessageBuilder;

        let mut model = GeneralNoiseModel::builder()
            .with_p_meas_crosstalk(1.0)
            .with_seed(42)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        // Prepare a bunch of |0> states
        builder.pz(&[0, 1, 2, 3, 4]);
        // Apply mid-circuit measurement and reset
        builder.mz(&[2]);
        builder.meas_crosstalk_global_payload(&[2]);
        builder.pz(&[2]);
        let _cmd = noise.start(builder.build()).unwrap();

        assert_eq!(
            noise.measured_qubits.len(),
            5,
            "There should be 5 measured qubits: one from MCMR and the others from
            crosstalk got: {:?}",
            noise.measured_qubits
        );

        let (q, gate_type) = noise.measured_qubits[0];
        assert_eq!(q, 2, "The first measurement should be the MCMR on qubit 2");
        assert!(
            !gate_type.is_crosstalk_payload(),
            "The first measurement should come from MCMR"
        );

        for (_, gate_type) in &noise.measured_qubits[1..] {
            assert!(
                gate_type.is_crosstalk_payload(),
                "The other measurements should come from crosstalk"
            );
        }

        // All results are 0
        let mut outcome_builder = ByteMessageBuilder::new();
        let _ = outcome_builder.for_outcomes();
        outcome_builder.add_outcomes(&[0, 0, 0, 0, 0]);

        let mcmr = noise.continue_processing(outcome_builder.build()).unwrap();

        let EngineStage::Complete(mcmr) = mcmr else {
            panic!("Expected Complete stage after processing outcomes");
        };
        let results = mcmr.outcomes().unwrap();

        assert_eq!(
            noise.measured_qubits.len(),
            0,
            "The list of measured_qubits should have been cleared."
        );
        assert_eq!(
            results.len(),
            1,
            "There should only be one outcome: that of the mid-circ measurement"
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
        // and we scale it by overall scale (2.0)
        let expected_leak_ratio = 0.5 * 2.0; // Base * overall scale, capped at 1.0

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
            .with_leakage_scale(0.5)
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
        // and we scale it by leakage_scale (0.5) and overall scale (2.0)
        let expected_leak_ratio = 0.01 * 2.0; // Base * overall scale

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

    #[test]
    fn test_p_idle_coherent() {
        // Create a circuit builder
        let mut builder = ByteMessage::quantum_operations_builder();

        // Create a noise model with coherent dephasing
        let mut model = GeneralNoiseModel::builder()
            .with_p_idle_coherent(true)
            .with_p_idle_quadratic_rate(0.2)
            .build();

        // Create an idle gate
        let gate = Gate {
            gate_type: GateType::Idle,
            angles: vec![].into(),
            qubits: vec![QubitId(0)].into(),
            params: vec![1.0].into(), // 1 second duration
        };

        // Apply idle faults - should use coherent dephasing (RZ gates)
        model.apply_idle_faults(&gate, 0.0, model.p_idle_quadratic_rate, &mut builder);

        // Get the message and verify it contains RZ gates
        let message = builder.build();
        let gates = message.quantum_ops().unwrap();

        // At least one gate should be an RZ gate
        assert!(!gates.is_empty(), "Should have at least one gate");
        assert!(
            gates.iter().any(|g| g.gate_type == GateType::RZ),
            "Should contain at least one RZ gate"
        );

        // Test multi-qubit idle gate
        let mut builder = ByteMessage::quantum_operations_builder();
        let multi_qubit_gate = Gate {
            gate_type: GateType::Idle,
            angles: vec![].into(),
            qubits: vec![QubitId(0), QubitId(1), QubitId(2)].into(), // 3 qubits
            params: vec![1.0].into(),                                // 1 second duration
        };

        model.apply_idle_faults(
            &multi_qubit_gate,
            0.0,
            model.p_idle_quadratic_rate,
            &mut builder,
        );

        let message = builder.build();
        let gates = message.quantum_ops().unwrap();

        // Should have a single RZ gate operating on multiple qubits
        assert!(
            !gates.is_empty(),
            "Should have at least one gate for multi-qubit idle"
        );
        let rz_gates: Vec<_> = gates
            .iter()
            .filter(|g| g.gate_type == GateType::RZ)
            .collect();
        assert_eq!(
            rz_gates.len(),
            1,
            "Should have 1 RZ gate for multi-qubit idle"
        );
        // The RZ gate should operate on all 3 qubits
        assert_eq!(
            rz_gates[0].qubits.len(),
            3,
            "RZ gate should operate on 3 qubits"
        );

        // Check that each qubit gets an RZ gate
        let mut affected_qubits: Vec<usize> = rz_gates
            .iter()
            .flat_map(|g| &g.qubits)
            .map(|&q| *q)
            .collect();
        affected_qubits.sort_unstable();
        assert_eq!(
            affected_qubits,
            vec![0, 1, 2],
            "RZ gates should affect qubits 0, 1, 2"
        );

        // Now test with incoherent dephasing
        let mut builder = ByteMessage::quantum_operations_builder();

        let mut model = GeneralNoiseModel::builder()
            .with_p_idle_coherent(false)
            .with_seed(42)
            .build();

        // Apply idle faults with incoherent dephasing
        model.apply_idle_faults(&gate, 0.0, model.p_idle_quadratic_rate, &mut builder);

        // The message may contain Z gates or be empty depending on random outcomes
        let message = builder.build();
        let _gates = message.quantum_ops().unwrap();

        // We can't assert specific outcomes due to randomness, but the code should run without errors
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_rzz_error_rate() {
        let mut model = GeneralNoiseModel::builder()
            .with_average_p2_probability(0.1)
            .with_p2_angle_params(0.1, 0.0, 0.25, 0.0)
            .with_p2_angle_power(1.0)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Test negative angle
        let neg_theta = -std::f64::consts::PI / 2.0;
        let error_neg = noise.p2_angle_error_rate(neg_theta);
        let expected_neg = 0.00625;
        assert!(
            (error_neg - expected_neg).abs() < 1e-6,
            "Expected {expected_neg}, got {error_neg}"
        );

        // Test positive angle
        let pos_theta = std::f64::consts::PI / 2.0;
        let error_pos = noise.p2_angle_error_rate(pos_theta);
        let expected_pos = 0.015625;
        assert!(
            (error_pos - expected_pos).abs() < 1e-6,
            "Expected {expected_pos}, got {error_pos}"
        );

        // Test quadratic scaling
        let mut model = GeneralNoiseModel::builder()
            .with_average_p2_probability(0.1)
            .with_p2_angle_params(0.1, 0.0, 0.25, 0.0)
            .with_p2_angle_power(2.0)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        let error_quad = noise.p2_angle_error_rate(pos_theta);
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
        let rz_gate = Gate {
            gate_type: GateType::RZ,
            angles: vec![Angle64::from_radians(0.1)].into(),
            qubits: vec![QubitId(0)].into(),
            params: vec![].into(),
        };

        // Create an X gate (not noiseless - should have noise applied)
        let x_gate = Gate {
            gate_type: GateType::X,
            angles: vec![].into(),
            qubits: vec![QubitId(0)].into(),
            params: vec![].into(),
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

        let mut builder = ByteMessage::quantum_operations_builder();
        builder.add_gate_commands(&[rz_gate.clone(), x_gate.clone()]);
        let msg = builder.build();

        // Apply noise to the gates manually since we can't access apply_noise_to_gates directly
        let state = noise.start(msg).unwrap();
        let EngineStage::NeedsProcessing(message) = state else {
            panic!("Expected NeedsProcessing stage after applying noise to gates");
        };
        let gates = message.quantum_ops().unwrap();

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
    fn test_leakage_scale() {
        // Create a noise model with leakage_scale set to 0.0
        let mut model = GeneralNoiseModel::builder().with_leakage_scale(0.0).build();
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
        assert!(!noise.is_leaked(0), "Qubit should not be marked as leaked");

        // Reset and try with leakage_scale 1.0
        let mut model = GeneralNoiseModel::builder().with_leakage_scale(1.0).build();
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
        assert!(noise.is_leaked(0), "Qubit should be marked as leaked");
    }

    #[test]
    fn test_rzz_error_rate_debug() {
        let mut model = GeneralNoiseModel::builder()
            .with_average_p2_probability(0.1)
            .with_p2_angle_params(0.1, 0.0, 0.25, 0.0)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        // Check unscaled przz error rate
        let theta = std::f64::consts::PI / 4.0;
        let norm_theta = theta / std::f64::consts::PI;
        let error_unscaled = noise.p2_angle_error_rate(theta);
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
            .with_p2_angle_params(0.1, 0.0, 0.25, 0.0)
            .with_scale(2.0)
            .build();
        let noise = model
            .as_any_mut()
            .downcast_mut::<GeneralNoiseModel>()
            .unwrap();

        let error_scaled = noise.p2_angle_error_rate(theta);

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
