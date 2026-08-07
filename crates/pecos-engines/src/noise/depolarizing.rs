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
use pecos_random::PecosRng;
use pecos_random::rng_ext::RngProbabilityExt;
use std::any::Any;

/////////////////////////////////////////////////////////
/// Tools for cataloging error opportunities and outcomes
/////////////////////////////////////////////////////////

/// The kinds of faults supported in cataloging
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepolarizingFaultSiteKind {
    Prep,
    Meas,
    SingleQubit,
    TwoQubit,
}

/// This class stores one possible outcome and its probability]
#[derive(Debug, Clone, PartialEq)]
pub struct DepolarizingFaultOutcome {
    /// Human-readable outcome label.
    pub label: &'static str,
    /// Outcome probability at this site.
    pub probability: f64,
}

/// Sites at which faults can occur
#[derive(Debug, Clone, PartialEq)]
pub struct DepolarizingFaultSite{
    // A globally unique identifier for this fault site
    pub uid: usize,
    pub gate_index: usize,
    pub kind: DepolarizingFaultSiteKind,
    pub gate_type: GateType,
    // Specify the error location
    pub qubits: Vec<usize>,
    // Specify possible states and their probabilities
    pub outcomes: Vec<DepolarizingFaultOutcome>,
}

// A fault catalog
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DepolarizingFaultCatalog {
    /// Ordered fault sites.
    pub sites: Vec<DepolarizingFaultSite>,
}

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
/// noise_model.set_seed(42); // For reproducibility
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
    /// Precomputed threshold for preparation error probability
    p_prep_threshold: u64,
    /// Precomputed threshold for measurement error probability
    p_meas_threshold: u64,
    /// Precomputed threshold for single-qubit gate error probability
    p1_threshold: u64,
    /// Precomputed threshold for two-qubit gate error probability
    p2_threshold: u64,
    /// Random number generator
    rng: NoiseRng<PecosRng>,
    /// Scratch builder reused across batches to avoid repeated allocations.
    scratch_builder: ByteMessageBuilder,
    /// Scratch gate storage reused across batches to avoid repeated allocations.
    scratch_gates: Vec<Gate>,
    /// If True, then all faults will be cataloged
    catalog_faults: bool,
    /// Stores the catalog of faults
    catalog: Option<DepolarizingFaultCatalog>,
}

impl ProbabilityValidator for DepolarizingNoiseModel {}

impl DepolarizingNoiseModel {
    fn channel_gate_error() -> PecosError {
        PecosError::Input(
            "ByteMessage noise models cannot process GateType::Channel; channel operations carry typed payloads and must use a channel-aware circuit path"
                .to_string(),
        )
    }

    /// Compute a probability threshold from a f64 probability
    #[inline]
    fn compute_threshold(p: f64) -> u64 {
        // Convert probability to fixed-point threshold for fast comparison
        // This matches the formula used in RngProbabilityExt::probability_threshold
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        {
            (p * (u64::MAX as f64)) as u64
        }
    }

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
            p_prep_threshold: Self::compute_threshold(p_prep),
            p_meas_threshold: Self::compute_threshold(p_meas),
            p1_threshold: Self::compute_threshold(p1),
            p2_threshold: Self::compute_threshold(p2),
            rng: NoiseRng::default(),
            scratch_builder: NoiseUtils::create_quantum_builder(),
            scratch_gates: Vec::new(),
            catalog_faults: false,
            catalog: None,
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

        // Recompute thresholds for optimized probability checks
        self.p_prep_threshold = Self::compute_threshold(p_prep);
        self.p_meas_threshold = Self::compute_threshold(p_meas);
        self.p1_threshold = Self::compute_threshold(p1);
        self.p2_threshold = Self::compute_threshold(p2);
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

    /// Enable or disable depolarizing fault-catalog capture during `start()`.
    pub fn set_catalog_faults_enabled(&mut self, enabled: bool) {
        self.catalog_faults = enabled;
    }

    /// Returns whether fault-catalog capture is enabled.
    #[must_use]
    pub fn catalog_faults_enabled(&self) -> bool {
        self.catalog_faults
    }

    /// Returns the catalog captured during `start()`
    #[must_use]
    pub fn fault_catalog(&self) -> Option<&DepolarizingFaultCatalog> {
        self.catalog.as_ref()
    }

    /// Takes ownership of the last captured catalog.
    pub fn take_fault_catalog(&mut self) -> Option<DepolarizingFaultCatalog> {
        self.catalog.take()
    }

    /// Build a fault catalog from a quantum-operation message.
    ///
    /// This does not modify the simulator state and is intended for
    /// pre-sampling catalog construction.
    ///
    /// # Errors
    /// Returns [`PecosError::Input`] if the message is not quantum-operations
    /// formatted.
    pub fn build_fault_catalog_from_message(
        &self,
        input: &ByteMessage,
    ) -> Result<DepolarizingFaultCatalog, PecosError> {

        // Convert message into vector of gate object
        let mut gates = Vec::new();
        input
            .quantum_ops_into(&mut gates)
            .map_err(|e| PecosError::Input(format!("Failed to parse quantum operations: {e}")))?;
    
        // Build up the fault catalog from the gates
        Ok(Self::build_fault_catalog_from_gates(
            &self,
            &gates,
        ))
    }


    /// Build a fault catalog from a vector of gates
    ///
    /// This does not modify the simulator state and is intended for
    /// pre-sampling catalog construction.
    fn build_fault_catalog_from_gates(
        &self,
        gates: &[Gate],
    ) -> DepolarizingFaultCatalog {

        // Create a vector to store the fault sites
        let mut fault_sites = Vec::new();

        // Track unique identifier for fault sites
        let mut next_fault_site_id = 0_usize;

        // Loop through provided gates
        for (gate_index, gate) in gates.iter().enumerate() {

            // Collect the qubits that the gate acts on
            let qubits = gate.qubits.iter().map(|q| q.0).collect::<Vec<_>>();

            // Collect gate kind and outcomes using matches of gate types
            let kind_outcomes = match gate.gate_type {
                // Single qubit gates
                GateType::X
                | GateType::Z
                | GateType::Y
                | GateType::SX
                | GateType::SXdg
                | GateType::SY
                | GateType::SYdg
                | GateType::SZ
                | GateType::SZdg
                | GateType::H
                | GateType::F
                | GateType::Fdg
                | GateType::RX
                | GateType::RY
                | GateType::RZ
                | GateType::T
                | GateType::Tdg
                | GateType::U
                | GateType::R1XY => Some((
                    DepolarizingFaultSiteKind::SingleQubit,
                    Self::single_qubit_outcomes(self.p1),
                )),
                // Two qubit gates
                GateType::CX
                | GateType::CY
                | GateType::CZ
                | GateType::CH
                | GateType::SXX
                | GateType::SXXdg
                | GateType::SYY
                | GateType::SYYdg
                | GateType::SZZ
                | GateType::SZZdg
                | GateType::SWAP
                | GateType::CRZ
                | GateType::RXX
                | GateType::RYY
                | GateType::RZZ
                | GateType::RXXRYYRZZ
                | GateType::U2q
                | GateType::CCX => Some((
                    DepolarizingFaultSiteKind::TwoQubit,
                    Self::two_qubit_outcomes(self.p2),
                )),
                // Measure 
                GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree => Some((
                    DepolarizingFaultSiteKind::Meas,
                    Self::binary_x_outcomes(self.p_meas),
                )),
                // Prepare
                GateType::PZ | GateType::QAlloc => Some((
                    DepolarizingFaultSiteKind::Prep,
                    Self::binary_x_outcomes(self.p_prep),
                )),
                // Gates that do not get a fault event
                GateType::Channel
                // TODO Should probably make sure identities are handled correctly
                | GateType::I
                | GateType::Idle
                | GateType::MeasCrosstalkLocalPayload
                | GateType::MeasCrosstalkGlobalPayload
                | GateType::QFree
                | GateType::Custom
                | GateType::TrackedPauliMeta => None,
            };

            // Add the new fault site
            if let Some((kind, outcomes)) = kind_outcomes {
                fault_sites.push(DepolarizingFaultSite {
                    uid: next_fault_site_id,
                    gate_index,
                    kind,
                    gate_type: gate.gate_type,
                    qubits,
                    outcomes,
                });

                // Increment the unique fault id
                next_fault_site_id += 1;
            }
        }

        DepolarizingFaultCatalog { sites: fault_sites }
    }

    fn binary_x_outcomes(p: f64) -> Vec<DepolarizingFaultOutcome> {
        vec![
            DepolarizingFaultOutcome {
                label: "NoFault",
                probability: 1.0 - p,
            },
            DepolarizingFaultOutcome {
                label: "X",
                probability: p,
            },
        ]
    }

    fn single_qubit_outcomes(p: f64) -> Vec<DepolarizingFaultOutcome> {
        let branch = p / 3.0;
        vec![
            DepolarizingFaultOutcome {
                label: "NoFault",
                probability: 1.0 - p,
            },
            DepolarizingFaultOutcome {
                label: "X",
                probability: branch,
            },
            DepolarizingFaultOutcome {
                label: "Y",
                probability: branch,
            },
            DepolarizingFaultOutcome {
                label: "Z",
                probability: branch,
            },
        ]
    }

    fn two_qubit_outcomes(p: f64) -> Vec<DepolarizingFaultOutcome> {
        const LABELS: [&str; 15] = [
            "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI",
            "ZX", "ZY", "ZZ",
        ];

        let mut outcomes = Vec::with_capacity(16);
        outcomes.push(DepolarizingFaultOutcome {
            label: "NoFault",
            probability: 1.0 - p,
        });

        let branch = p / 15.0;
        for label in LABELS {
            outcomes.push(DepolarizingFaultOutcome {
                label,
                probability: branch,
            });
        }

        outcomes
    }

    fn apply_noise_to_gate(
        rng: &mut NoiseRng<PecosRng>,
        p_prep_threshold: u64,
        p_meas_threshold: u64,
        p1_threshold: u64,
        p2_threshold: u64,
        builder: &mut ByteMessageBuilder,
        gate: &Gate,
    ) {
        match gate.gate_type {
            GateType::X
            | GateType::Z
            | GateType::Y
            | GateType::SX
            | GateType::SXdg
            | GateType::SY
            | GateType::SYdg
            | GateType::SZ
            | GateType::SZdg
            | GateType::H
            | GateType::F
            | GateType::Fdg
            | GateType::RX
            | GateType::RY
            | GateType::RZ
            | GateType::T
            | GateType::Tdg
            | GateType::U
            | GateType::R1XY => {
                NoiseUtils::add_gate_to_builder(builder, gate);
                trace!("Applying single-qubit gate with possible fault");
                Self::apply_sq_faults(rng, p1_threshold, builder, gate);
            }
            GateType::CX
            | GateType::CY
            | GateType::CZ
            | GateType::CH
            | GateType::SXX
            | GateType::SXXdg
            | GateType::SYY
            | GateType::SYYdg
            | GateType::SZZ
            | GateType::SZZdg
            | GateType::SWAP
            | GateType::CRZ
            | GateType::RXX
            | GateType::RYY
            | GateType::RZZ
            | GateType::RXXRYYRZZ
            | GateType::U2q => {
                NoiseUtils::add_gate_to_builder(builder, gate);
                trace!("Applying two-qubit gate with possible fault");
                Self::apply_tq_faults(rng, p2_threshold, builder, gate);
            }
            GateType::CCX => {
                NoiseUtils::add_gate_to_builder(builder, gate);
                trace!("Applying three-qubit gate with possible fault");
                Self::apply_tq_faults(rng, p2_threshold, builder, gate);
            }
            GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree => {
                trace!("Applying measurement with possible fault");
                Self::apply_meas_faults(rng, p_meas_threshold, builder, gate);
                NoiseUtils::add_gate_to_builder(builder, gate);
            }
            GateType::PZ | GateType::QAlloc => {
                NoiseUtils::add_gate_to_builder(builder, gate);
                trace!("Applying preparation with possible fault");
                Self::apply_prep_faults(rng, p_prep_threshold, builder, gate);
            }
            GateType::Channel => unreachable!("channel gates are rejected before noise is applied"),
            GateType::I
            | GateType::Idle
            | GateType::MeasCrosstalkLocalPayload
            | GateType::MeasCrosstalkGlobalPayload
            | GateType::QFree
            | GateType::Custom
            | GateType::TrackedPauliMeta => {
                // Just pass through with no added noise.
            }
        }
    }

    fn apply_prep_faults(
        rng: &mut NoiseRng<PecosRng>,
        p_prep_threshold: u64,
        builder: &mut ByteMessageBuilder,
        gate: &Gate,
    ) {
        // Use precomputed threshold for fast probability check
        if rng.inner_mut().check_probability(p_prep_threshold) {
            trace!("Applying prep fault on qubits {:?}", gate.qubits);
            NoiseUtils::apply_x(builder, *gate.qubits[0]);
        }
    }

    fn apply_meas_faults(
        rng: &mut NoiseRng<PecosRng>,
        p_meas_threshold: u64,
        builder: &mut ByteMessageBuilder,
        gate: &Gate,
    ) {
        // Use precomputed threshold for fast probability check
        if rng.inner_mut().check_probability(p_meas_threshold) {
            trace!("Applying meas fault on qubits {:?}", gate.qubits);
            NoiseUtils::apply_x(builder, *gate.qubits[0]);
        }
    }

    fn apply_sq_faults(
        rng: &mut NoiseRng<PecosRng>,
        p1_threshold: u64,
        builder: &mut ByteMessageBuilder,
        gate: &Gate,
    ) {
        // Use fused noise sampling: probability check + Pauli selection in one call
        if let Some(fault_type) = rng.inner_mut().noise_sample_1q(p1_threshold) {
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

    fn apply_tq_faults(
        rng: &mut NoiseRng<PecosRng>,
        p2_threshold: u64,
        builder: &mut ByteMessageBuilder,
        gate: &Gate,
    ) {
        // Use fused noise sampling: probability check + Pauli selection in one call
        if let Some(fault_type) = rng.inner_mut().noise_sample_2q(p2_threshold) {
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
    type Rng = PecosRng;

    fn set_rng(&mut self, rng: PecosRng) {
        self.rng = NoiseRng::new(rng);
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
            noise.set_seed(seed);
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

        self.scratch_gates.clear();
        input
            .quantum_ops_into(&mut self.scratch_gates)
            .map_err(|e| PecosError::Input(format!("Failed to parse quantum operations: {e}")))?;

        if self.scratch_gates.iter().any(Gate::is_channel) {
            return Err(Self::channel_gate_error());
        }

        // Initialize an empty fault catalog
        if self.catalog_faults {
            self.catalog = Some(Self::build_fault_catalog_from_gates(
                &self,
                &self.scratch_gates,
            ));
        } else {
            self.catalog = None;
        }

        if self.p_prep_threshold == 0
            && self.p_meas_threshold == 0
            && self.p1_threshold == 0
            && self.p2_threshold == 0
        {
            return Ok(EngineStage::NeedsProcessing(input));
        }

        self.scratch_builder.reset();
        let _ = self.scratch_builder.for_quantum_operations();

        let p_prep_threshold = self.p_prep_threshold;
        let p_meas_threshold = self.p_meas_threshold;
        let p1_threshold = self.p1_threshold;
        let p2_threshold = self.p2_threshold;
        let rng = &mut self.rng;
        let builder = &mut self.scratch_builder;

        for gate in &self.scratch_gates {
            Self::apply_noise_to_gate(
                rng,
                p_prep_threshold,
                p_meas_threshold,
                p1_threshold,
                p2_threshold,
                builder,
                gate,
            );
        }

        let noisy_gates = self.scratch_builder.build();

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
        self.catalog = None;
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
        builder.x(&[0]);
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
        builder.x(&[0]);
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
    fn test_fault_catalog_from_message_basic_attributes() {
        // Check that the catalog has the correct information stored
        // in it from a very simple circuit
        let noise = DepolarizingNoiseModel::new(0.1, 0.2, 0.3, 0.4);

        // Build a simple circuit
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.pz(&[0]);
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[1]);

        let msg = builder.build();

        // Try to build the catalog
        let catalog = noise
            .build_fault_catalog_from_message(&msg)
            .expect("catalog generation should succeed");

        // Check that the catalog has the correct information stored
        assert_eq!(catalog.sites.len(), 4);

        // Check that the sites have the correct kind
        assert_eq!(catalog.sites[0].kind, DepolarizingFaultSiteKind::Prep);
        assert_eq!(catalog.sites[1].kind, DepolarizingFaultSiteKind::SingleQubit);
        assert_eq!(catalog.sites[2].kind, DepolarizingFaultSiteKind::TwoQubit);
        assert_eq!(catalog.sites[3].kind, DepolarizingFaultSiteKind::Meas);

        // Check that the sites have the correct unique ids
        assert_eq!(catalog.sites[0].uid, 0);
        assert_eq!(catalog.sites[1].uid, 1);
        assert_eq!(catalog.sites[2].uid, 2);
        assert_eq!(catalog.sites[3].uid, 3);

        // Check that the gates have been indexed correct
        assert_eq!(catalog.sites[0].gate_index, 0);
        assert_eq!(catalog.sites[1].gate_index, 1);
        assert_eq!(catalog.sites[2].gate_index, 2);
        assert_eq!(catalog.sites[3].gate_index, 3);

        // Check that the gate types have been indexed correctly
        assert_eq!(catalog.sites[0].gate_type, GateType::PZ);
        assert_eq!(catalog.sites[1].gate_type, GateType::X);
        assert_eq!(catalog.sites[2].gate_type, GateType::CX);
        assert_eq!(catalog.sites[3].gate_type, GateType::MZ);

        // Check that the qubits are stored correctly
        assert_eq!(catalog.sites[0].qubits, vec![0]);
        assert_eq!(catalog.sites[1].qubits, vec![0]);
        assert_eq!(catalog.sites[2].qubits, vec![0, 1]);
        assert_eq!(catalog.sites[3].qubits, vec![1]);

        // Check that the outcomes are correct
        assert_eq!(catalog.sites[0].outcomes.len(), 2);
        assert_eq!(catalog.sites[1].outcomes.len(), 4);
        assert_eq!(catalog.sites[2].outcomes.len(), 16);
        assert_eq!(catalog.sites[3].outcomes.len(), 2);
        
        assert_eq!(catalog.sites[0].outcomes[0].label, "NoFault");
        assert_eq!(catalog.sites[0].outcomes[1].label, "X");

        assert_eq!(catalog.sites[1].outcomes[0].label, "NoFault");
        assert_eq!(catalog.sites[1].outcomes[1].label, "X");
        assert_eq!(catalog.sites[1].outcomes[2].label, "Y");
        assert_eq!(catalog.sites[1].outcomes[3].label, "Z");

        assert_eq!(catalog.sites[2].outcomes[0].label, "NoFault");
        assert_eq!(catalog.sites[2].outcomes[1].label, "IX");
        assert_eq!(catalog.sites[2].outcomes[2].label, "IY");
        assert_eq!(catalog.sites[2].outcomes[3].label, "IZ");
        assert_eq!(catalog.sites[2].outcomes[4].label, "XI");
        assert_eq!(catalog.sites[2].outcomes[5].label, "XX");
        assert_eq!(catalog.sites[2].outcomes[6].label, "XY");
        assert_eq!(catalog.sites[2].outcomes[7].label, "XZ");
        assert_eq!(catalog.sites[2].outcomes[8].label, "YI");
        assert_eq!(catalog.sites[2].outcomes[9].label, "YX");
        assert_eq!(catalog.sites[2].outcomes[10].label, "YY");
        assert_eq!(catalog.sites[2].outcomes[11].label, "YZ");
        assert_eq!(catalog.sites[2].outcomes[12].label, "ZI");
        assert_eq!(catalog.sites[2].outcomes[13].label, "ZX");
        assert_eq!(catalog.sites[2].outcomes[14].label, "ZY");
        assert_eq!(catalog.sites[2].outcomes[15].label, "ZZ");

    }

    #[test]
    fn test_fault_catalog_probabilities_sum_to_one() {
        // Checks that all the fault catalog outcomes
        // sum to 1
        let noise = DepolarizingNoiseModel::new(0.1, 0.2, 0.3, 0.45);

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[1]);
        builder.pz(&[2]);

        let msg = builder.build();
        let catalog = noise
            .build_fault_catalog_from_message(&msg)
            .expect("catalog generation should succeed");

        for site in &catalog.sites {
            let sum: f64 = site.outcomes.iter().map(|o| o.probability).sum();
            assert!((sum - 1.0).abs() < 1e-12, "outcomes must sum to one");
            // Check that the first outcome is always NoFault
            assert_eq!(site.outcomes[0].label, "NoFault");
        }
    }

    #[test]
    fn test_fault_catalog_capture_in_start() {
        // 
        let mut noise = DepolarizingNoiseModel::new_uniform(0.0);
        noise.set_catalog_faults_enabled(true);

        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        builder.x(&[0]);
        builder.cx(&[(0, 1)]);
        let msg = builder.build();

        let _ = noise.start(msg).expect("noise start should succeed");

        let catalog = noise
            .fault_catalog()
            .expect("catalog should be captured when enabled");
        assert_eq!(catalog.sites.len(), 2);
    }
}
