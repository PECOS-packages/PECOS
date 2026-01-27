// Copyright 2026 The PECOS Developers
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

//! Gadget-level fault tolerance checking.
//!
//! QEC protocols are composed of gadgets - circuits that may have input qubits
//! (carrying errors from previous stages) and output qubits (passing errors to
//! next stages). For t-fault tolerant gadgets, we need:
//!
//! **s + r <= t** where s = input fault weight, r = internal fault weight
//!
//! # Gadget Types
//!
//! | Type | Input | Output | Example |
//! |------|-------|--------|---------|
//! | State Preparation | None | Data qubits | Prepare |0_L⟩ |
//! | Syndrome Extraction | Data qubits | Data qubits + syndrome | EC round |
//! | Measurement | Data qubits | None (all measured) | Final readout |
//! | Gate | Data qubits | Data qubits | Logical CNOT |
//! | Self-contained | None | None | Full QEC experiment |
//!
//! # Usage
//!
//! ```
//! use pecos_qec::fault_tolerance::{GadgetConfig, GadgetChecker};
//! use pecos_quantum::TickCircuit;
//!
//! // Build a syndrome extraction gadget
//! let mut circuit = TickCircuit::new();
//! // Data qubits 0,1,2 are INPUT (not initialized here)
//! circuit.tick().pz(&[3, 4]);           // Initialize ancillas only
//! circuit.tick().cx(&[(0, 3), (1, 4)]); // CNOTs from data to ancilla
//! circuit.tick().cx(&[(1, 3), (2, 4)]);
//! circuit.tick().mz(&[3, 4]);           // Measure ancillas
//! // Data qubits 0,1,2 are OUTPUT (not measured here)
//!
//! let config = GadgetConfig::syndrome_extraction()
//!     .with_input_qubits(&[0, 1, 2])   // Data qubits enter with potential errors
//!     .with_output_qubits(&[0, 1, 2])  // Data qubits leave (may have errors)
//!     .with_ancilla_qubits(&[3, 4])    // Initialized and measured within gadget
//!     .with_z_ancillas(&[3, 4])        // For syndrome extraction
//!     .with_logical_z(&[], &[0, 1, 2]); // Z logical operator
//!
//! let checker = GadgetChecker::new(&circuit, config);
//! let analysis = checker.analyze(1); // Check 1-fault tolerance
//!
//! println!("Is 1-FT: {}", analysis.is_fault_tolerant());
//! ```

use super::pauli_prop_checker::{
    CircuitIO, MeasurementRound, SyndromeClass, compute_stabilizer_syndromes,
    extract_measurement_rounds, extract_output_error,
};
use super::{
    FaultCheckConfig, FaultConfiguration, PauliFaultIterator, SpacetimeLocation,
    extract_spacetime_locations, propagate_faults,
};
use pecos_core::QubitId;
use pecos_qsim::{CliffordGateable, PauliProp};
use pecos_quantum::TickCircuit;
use std::collections::{BTreeSet, HashMap};

/// Configuration for gadget-level fault tolerance checking.
#[derive(Debug, Clone)]
pub struct GadgetConfig {
    /// Qubits that enter the gadget (not initialized within it).
    /// These may carry errors from previous stages.
    pub input_qubits: Vec<usize>,

    /// Qubits that exit the gadget (not measured within it).
    /// Errors on these propagate to subsequent stages.
    pub output_qubits: Vec<usize>,

    /// Ancilla qubits (initialized and measured within the gadget).
    pub ancilla_qubits: Vec<usize>,

    /// Z-basis measurement qubits (for detecting X errors).
    pub z_ancillas: Vec<usize>,

    /// X-basis measurement qubits (for detecting Z errors).
    pub x_ancillas: Vec<usize>,

    /// Logical Z operators as (X positions, Z positions) pairs.
    pub logical_zs: Vec<(Vec<usize>, Vec<usize>)>,

    /// Logical X operators as (X positions, Z positions) pairs.
    pub logical_xs: Vec<(Vec<usize>, Vec<usize>)>,

    /// Pauli types to include in analysis.
    pub include_x: bool,
    pub include_y: bool,
    pub include_z: bool,
}

impl Default for GadgetConfig {
    fn default() -> Self {
        Self {
            input_qubits: Vec::new(),
            output_qubits: Vec::new(),
            ancilla_qubits: Vec::new(),
            z_ancillas: Vec::new(),
            x_ancillas: Vec::new(),
            logical_zs: Vec::new(),
            logical_xs: Vec::new(),
            include_x: true,
            include_y: true,
            include_z: true,
        }
    }
}

impl GadgetConfig {
    /// Creates a new empty gadget configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a configuration for a state preparation gadget.
    ///
    /// State preparation has no input qubits (all initialized) but has output qubits.
    #[must_use]
    pub fn state_preparation() -> Self {
        Self::default()
    }

    /// Creates a configuration for a syndrome extraction gadget.
    ///
    /// Syndrome extraction has both input and output data qubits.
    #[must_use]
    pub fn syndrome_extraction() -> Self {
        Self::default()
    }

    /// Creates a configuration for a measurement gadget.
    ///
    /// Measurement has input qubits but no output qubits (all measured).
    #[must_use]
    pub fn measurement() -> Self {
        Self::default()
    }

    /// Creates a configuration for a self-contained circuit.
    ///
    /// Self-contained circuits have no input or output qubits.
    #[must_use]
    pub fn self_contained() -> Self {
        Self::default()
    }

    /// Creates a configuration with I/O auto-detected from the circuit.
    ///
    /// This analyzes the circuit structure to determine:
    /// - Input qubits: used but never prepared (carry data from previous stage)
    /// - Output qubits: used but never measured (carry data to next stage)
    /// - Ancilla qubits: prepared within the circuit
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_qec::fault_tolerance::GadgetConfig;
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().pz(&[3, 4]);           // Ancillas prepared
    /// circuit.tick().cx(&[(0, 3), (1, 4)]); // Data qubits used
    /// circuit.tick().mz(&[3, 4]);           // Ancillas measured
    ///
    /// let config = GadgetConfig::from_circuit(&circuit);
    /// assert!(config.has_input());  // Data qubits 0, 1 are inputs
    /// assert!(config.has_output()); // Data qubits 0, 1 are outputs
    /// ```
    #[must_use]
    pub fn from_circuit(circuit: &TickCircuit) -> Self {
        let io = CircuitIO::from_circuit(circuit);
        Self {
            input_qubits: io.input_qubits,
            output_qubits: io.output_qubits,
            ancilla_qubits: io.ancilla_qubits,
            ..Self::default()
        }
    }

    /// Populates I/O configuration by detecting from a circuit.
    ///
    /// This analyzes the circuit structure to determine input, output, and ancilla qubits.
    #[must_use]
    pub fn with_detected_io(mut self, circuit: &TickCircuit) -> Self {
        let io = CircuitIO::from_circuit(circuit);
        self.input_qubits = io.input_qubits;
        self.output_qubits = io.output_qubits;
        self.ancilla_qubits = io.ancilla_qubits;
        self
    }

    /// Sets the input qubits (qubits entering the gadget with potential errors).
    #[must_use]
    pub fn with_input_qubits(mut self, qubits: &[usize]) -> Self {
        self.input_qubits = qubits.to_vec();
        self
    }

    /// Sets the output qubits (qubits exiting the gadget).
    #[must_use]
    pub fn with_output_qubits(mut self, qubits: &[usize]) -> Self {
        self.output_qubits = qubits.to_vec();
        self
    }

    /// Sets the ancilla qubits (initialized and measured within gadget).
    #[must_use]
    pub fn with_ancilla_qubits(mut self, qubits: &[usize]) -> Self {
        self.ancilla_qubits = qubits.to_vec();
        self
    }

    /// Sets the Z-basis measurement qubits.
    #[must_use]
    pub fn with_z_ancillas(mut self, qubits: &[usize]) -> Self {
        self.z_ancillas = qubits.to_vec();
        self
    }

    /// Sets the X-basis measurement qubits.
    #[must_use]
    pub fn with_x_ancillas(mut self, qubits: &[usize]) -> Self {
        self.x_ancillas = qubits.to_vec();
        self
    }

    /// Adds a logical Z operator.
    #[must_use]
    pub fn with_logical_z(mut self, x_positions: &[usize], z_positions: &[usize]) -> Self {
        self.logical_zs
            .push((x_positions.to_vec(), z_positions.to_vec()));
        self
    }

    /// Adds a logical X operator.
    #[must_use]
    pub fn with_logical_x(mut self, x_positions: &[usize], z_positions: &[usize]) -> Self {
        self.logical_xs
            .push((x_positions.to_vec(), z_positions.to_vec()));
        self
    }

    /// Configures to only check X-type errors (CSS mode for Z-distance).
    #[must_use]
    pub fn x_only(mut self) -> Self {
        self.include_x = true;
        self.include_y = false;
        self.include_z = false;
        self
    }

    /// Configures to only check Z-type errors (CSS mode for X-distance).
    #[must_use]
    pub fn z_only(mut self) -> Self {
        self.include_x = false;
        self.include_y = false;
        self.include_z = true;
        self
    }

    /// Returns the Pauli types to include.
    #[must_use]
    pub fn pauli_types(&self) -> Vec<u8> {
        let mut types = Vec::with_capacity(3);
        if self.include_x {
            types.push(1);
        }
        if self.include_y {
            types.push(2);
        }
        if self.include_z {
            types.push(3);
        }
        types
    }

    /// Returns true if this gadget has input qubits.
    #[must_use]
    pub fn has_input(&self) -> bool {
        !self.input_qubits.is_empty()
    }

    /// Returns true if this gadget has output qubits.
    #[must_use]
    pub fn has_output(&self) -> bool {
        !self.output_qubits.is_empty()
    }
}

/// Result of analyzing a single fault combination in a gadget.
#[derive(Debug, Clone)]
pub struct GadgetFaultResult {
    /// Weight of input faults (errors on input qubits).
    pub input_weight: usize,

    /// Weight of internal faults (errors during gadget execution).
    pub internal_weight: usize,

    /// The input fault configuration (errors on input qubits at start).
    pub input_faults: Vec<(usize, u8)>, // (qubit, pauli_type)

    /// The internal fault configuration.
    pub internal_faults: FaultConfiguration,

    /// The propagated error state at end of gadget.
    pub propagated_error: PauliProp,

    /// Z syndrome flips (which Z-basis measurements would flip).
    pub z_syndrome_flips: Vec<usize>,

    /// X syndrome flips (which X-basis measurements would flip).
    pub x_syndrome_flips: Vec<usize>,

    /// Whether each logical operator is flipped.
    pub logical_errors: Vec<bool>,

    /// Weight of residual error on output qubits.
    pub output_error_weight: usize,

    /// Classification of this fault combination.
    pub classification: GadgetFaultClass,
}

/// Classification of a fault combination in a gadget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GadgetFaultClass {
    /// No effect - equivalent to identity or stabilizer.
    Harmless,

    /// Detectable via syndrome, no logical error, bounded output weight.
    /// A good decoder should handle this.
    Correctable,

    /// Detectable but causes logical error.
    /// Detected but may exceed correction capacity.
    DetectedLogicalError,

    /// Undetectable logical error - fatal.
    /// No syndrome but causes logical failure.
    UndetectedLogicalError,

    /// Output error weight exceeds threshold.
    /// Even without logical error, too much damage for next stage.
    ExcessiveOutputError,
}

impl GadgetFaultClass {
    /// Returns true if this is a fault tolerance failure.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        matches!(
            self,
            GadgetFaultClass::UndetectedLogicalError | GadgetFaultClass::ExcessiveOutputError
        )
    }
}

/// Analysis result for gadget-level fault tolerance.
#[derive(Debug, Clone)]
pub struct GadgetAnalysis {
    /// Maximum fault weight tested.
    pub max_weight: usize,

    /// Total number of fault combinations tested.
    pub total_tested: usize,

    /// Number of harmless fault combinations.
    pub harmless: usize,

    /// Number of correctable fault combinations.
    pub correctable: usize,

    /// Number of detected logical errors.
    pub detected_logical: usize,

    /// Number of undetected logical errors (fatal).
    pub undetected_logical: usize,

    /// Number of excessive output error combinations.
    pub excessive_output: usize,

    /// Detailed results for failures (if collected).
    pub failures: Vec<GadgetFaultResult>,
}

impl GadgetAnalysis {
    /// Returns true if the gadget is fault-tolerant to the tested weight.
    ///
    /// A gadget is t-FT if no combination of s input faults + r internal faults
    /// with s + r <= t causes an undetected logical error or excessive output error.
    #[must_use]
    pub fn is_fault_tolerant(&self) -> bool {
        self.undetected_logical == 0 && self.excessive_output == 0
    }

    /// Returns the number of failures.
    #[must_use]
    pub fn num_failures(&self) -> usize {
        self.undetected_logical + self.excessive_output
    }

    /// Returns a summary of failure modes.
    #[must_use]
    pub fn failure_summary(&self) -> Vec<String> {
        let mut summary = Vec::new();
        if self.undetected_logical > 0 {
            summary.push(format!(
                "{} undetected logical errors",
                self.undetected_logical
            ));
        }
        if self.excessive_output > 0 {
            summary.push(format!(
                "{} excessive output error combinations",
                self.excessive_output
            ));
        }
        summary
    }
}

// ============================================================================
// Decoder Analysis Types
// ============================================================================

/// Analysis result for decoder requirements at the gadget level.
///
/// This groups faults by their syndrome pattern and classifies each pattern
/// as correctable, detected-uncorrectable, or ambiguous. Ambiguous syndromes
/// indicate a fundamental decoder limitation where the syndrome alone cannot
/// determine the correct recovery.
#[derive(Debug, Clone)]
pub struct GadgetDecoderAnalysis {
    /// Maximum fault weight tested.
    pub max_weight: usize,

    /// Total number of fault combinations tested.
    pub total_tested: usize,

    /// Detailed analysis for each non-trivial syndrome pattern.
    pub syndromes: Vec<GadgetSyndromeAnalysis>,

    /// Number of syndrome patterns where all faults are correctable.
    pub correctable_syndromes: usize,

    /// Number of syndrome patterns where all faults cause logical errors.
    pub detected_uncorrectable_syndromes: usize,

    /// Number of syndrome patterns with mixed outcomes (decoder must guess).
    pub ambiguous_syndromes: usize,

    /// Number of faults with no syndrome that cause logical errors (fatal).
    pub undetectable_logical_errors: usize,

    /// Number of faults with no syndrome that are harmless (stabilizer equivalent).
    pub undetectable_stabilizers: usize,
}

impl GadgetDecoderAnalysis {
    /// Returns true if the gadget is fault-tolerant from a decoder perspective.
    ///
    /// A gadget is decoder-FT if there are no undetectable logical errors
    /// and no ambiguous syndromes (where decoder must guess).
    #[must_use]
    pub fn is_fault_tolerant(&self) -> bool {
        self.undetectable_logical_errors == 0 && self.ambiguous_syndromes == 0
    }

    /// Returns the total number of distinct syndrome patterns.
    #[must_use]
    pub fn num_syndrome_patterns(&self) -> usize {
        self.syndromes.len()
    }

    /// Returns syndromes that are problematic (ambiguous or detected-uncorrectable).
    #[must_use]
    pub fn problematic_syndromes(&self) -> Vec<&GadgetSyndromeAnalysis> {
        self.syndromes
            .iter()
            .filter(|s| !matches!(s.class, SyndromeClass::Correctable))
            .collect()
    }
}

/// Analysis of a single syndrome pattern.
#[derive(Debug, Clone)]
pub struct GadgetSyndromeAnalysis {
    /// The syndrome pattern (Z flips then X flips, or combined).
    pub syndrome: Vec<u8>,

    /// Number of faults with this syndrome that cause no logical error.
    pub correctable_count: usize,

    /// Number of faults with this syndrome that cause a logical error.
    pub uncorrectable_count: usize,

    /// Classification of this syndrome pattern.
    pub class: SyndromeClass,
}

// ============================================================================
// Follow-Up Analysis Types
// ============================================================================

/// Configuration for follow-up syndrome analysis.
///
/// For gadgets with output qubits, the gadget's own syndrome may be ambiguous.
/// By considering what syndrome a hypothetical ideal EC round following the
/// gadget would produce, we can often disambiguate.
#[derive(Debug, Clone)]
pub struct GadgetFollowUpConfig {
    /// Stabilizers that would be measured in an ideal EC round after the gadget.
    /// Each stabilizer is (X positions, Z positions).
    pub follow_up_stabilizers: Vec<(Vec<usize>, Vec<usize>)>,
}

impl GadgetFollowUpConfig {
    /// Creates a new follow-up config with the given stabilizers.
    #[must_use]
    pub fn new(stabilizers: Vec<(Vec<usize>, Vec<usize>)>) -> Self {
        Self {
            follow_up_stabilizers: stabilizers,
        }
    }

    /// Builder method to add a stabilizer.
    #[must_use]
    pub fn with_stabilizer(mut self, x_positions: &[usize], z_positions: &[usize]) -> Self {
        self.follow_up_stabilizers
            .push((x_positions.to_vec(), z_positions.to_vec()));
        self
    }
}

// ============================================================================
// Syndrome History Analysis Types
// ============================================================================

/// Analysis result considering full syndrome history across measurement rounds.
///
/// For gadgets with multiple measurement rounds, this tracks the complete
/// syndrome evolution rather than just the final syndrome.
#[derive(Debug, Clone)]
pub struct GadgetHistoryAnalysis {
    /// The measurement rounds detected in the circuit.
    pub rounds: Vec<MeasurementRound>,

    /// Distinct syndrome history patterns and their analysis.
    pub histories: Vec<GadgetHistoryPattern>,

    /// Number of history patterns where all faults are correctable.
    pub correctable_histories: usize,

    /// Number of history patterns where all faults cause logical errors.
    pub uncorrectable_histories: usize,

    /// Number of history patterns with mixed outcomes.
    pub ambiguous_histories: usize,

    /// Total fault combinations tested.
    pub total_tested: usize,

    /// Faults with no syndrome in any round that cause logical errors.
    pub never_detected_logical_errors: usize,

    /// Faults with no syndrome in any round that are harmless.
    pub never_detected_stabilizers: usize,
}

impl GadgetHistoryAnalysis {
    /// Returns true if the gadget is fault-tolerant considering syndrome history.
    #[must_use]
    pub fn is_fault_tolerant(&self) -> bool {
        self.never_detected_logical_errors == 0 && self.ambiguous_histories == 0
    }
}

/// Analysis of a single syndrome history pattern.
#[derive(Debug, Clone)]
pub struct GadgetHistoryPattern {
    /// The complete syndrome history (syndrome vector at each measurement round).
    pub history: Vec<Vec<u8>>,

    /// Number of faults with this history that cause no logical error.
    pub correctable_count: usize,

    /// Number of faults with this history that cause a logical error.
    pub uncorrectable_count: usize,

    /// Classification of this history pattern.
    pub class: SyndromeClass,
}

/// Gadget-level fault tolerance checker.
///
/// This checker handles gadgets with input/output qubits, enumerating all
/// combinations of input faults (s) and internal faults (r) with s + r <= t.
pub struct GadgetChecker<'a> {
    circuit: &'a TickCircuit,
    config: GadgetConfig,
    internal_locations: Vec<SpacetimeLocation>,
}

impl<'a> GadgetChecker<'a> {
    /// Creates a new gadget checker.
    #[must_use]
    pub fn new(circuit: &'a TickCircuit, config: GadgetConfig) -> Self {
        // Extract internal fault locations (excluding input qubit initialization)
        let internal_locations = extract_spacetime_locations(circuit, false);

        Self {
            circuit,
            config,
            internal_locations,
        }
    }

    /// Creates a new gadget checker with I/O auto-detected from the circuit.
    ///
    /// This is a convenience constructor that analyzes the circuit structure
    /// to determine input, output, and ancilla qubits automatically.
    ///
    /// Note: You still need to specify syndrome qubits and logical operators
    /// using the returned checker's config.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_qec::fault_tolerance::GadgetChecker;
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().pz(&[3, 4]);
    /// circuit.tick().cx(&[(0, 3), (1, 4)]);
    /// circuit.tick().mz(&[3, 4]);
    ///
    /// let checker = GadgetChecker::from_circuit(&circuit)
    ///     .with_z_ancillas(&[3, 4])
    ///     .with_logical_z(&[], &[0, 1]);
    ///
    /// assert!(checker.has_input_qubits());
    /// ```
    #[must_use]
    pub fn from_circuit(circuit: &'a TickCircuit) -> Self {
        let config = GadgetConfig::from_circuit(circuit);
        Self::new(circuit, config)
    }

    /// Returns true if this gadget has input qubits.
    #[must_use]
    pub fn has_input_qubits(&self) -> bool {
        self.config.has_input()
    }

    /// Returns true if this gadget has output qubits.
    #[must_use]
    pub fn has_output_qubits(&self) -> bool {
        self.config.has_output()
    }

    /// Returns the input qubits (enter the gadget with potential errors).
    #[must_use]
    pub fn input_qubits(&self) -> &[usize] {
        &self.config.input_qubits
    }

    /// Returns the output qubits (exit the gadget, may carry errors).
    #[must_use]
    pub fn output_qubits(&self) -> &[usize] {
        &self.config.output_qubits
    }

    /// Returns the ancilla qubits (initialized within the gadget).
    #[must_use]
    pub fn ancilla_qubits(&self) -> &[usize] {
        &self.config.ancilla_qubits
    }

    /// Returns the gadget configuration.
    #[must_use]
    pub fn config(&self) -> &GadgetConfig {
        &self.config
    }

    /// Sets the Z-basis measurement qubits.
    #[must_use]
    pub fn with_z_ancillas(mut self, qubits: &[usize]) -> Self {
        self.config.z_ancillas = qubits.to_vec();
        self
    }

    /// Sets the X-basis measurement qubits.
    #[must_use]
    pub fn with_x_ancillas(mut self, qubits: &[usize]) -> Self {
        self.config.x_ancillas = qubits.to_vec();
        self
    }

    /// Adds a logical Z operator.
    #[must_use]
    pub fn with_logical_z(mut self, x_positions: &[usize], z_positions: &[usize]) -> Self {
        self.config
            .logical_zs
            .push((x_positions.to_vec(), z_positions.to_vec()));
        self
    }

    /// Adds a logical X operator.
    #[must_use]
    pub fn with_logical_x(mut self, x_positions: &[usize], z_positions: &[usize]) -> Self {
        self.config
            .logical_xs
            .push((x_positions.to_vec(), z_positions.to_vec()));
        self
    }

    /// Analyzes fault tolerance up to the given weight.
    ///
    /// For gadgets with input qubits, this enumerates all (s, r) combinations
    /// where s = input fault weight and r = internal fault weight, with s + r <= t.
    #[must_use]
    pub fn analyze(&self, max_weight: usize) -> GadgetAnalysis {
        self.analyze_with_options(max_weight, true)
    }

    /// Analyzes fault tolerance with options.
    ///
    /// # Arguments
    /// * `max_weight` - Maximum total fault weight (input + internal)
    /// * `collect_failures` - Whether to store detailed failure information
    #[must_use]
    pub fn analyze_with_options(
        &self,
        max_weight: usize,
        collect_failures: bool,
    ) -> GadgetAnalysis {
        let mut analysis = GadgetAnalysis {
            max_weight,
            total_tested: 0,
            harmless: 0,
            correctable: 0,
            detected_logical: 0,
            undetected_logical: 0,
            excessive_output: 0,
            failures: Vec::new(),
        };

        // Enumerate all (input_weight, internal_weight) combinations
        for input_weight in 0..=max_weight {
            let max_internal = max_weight - input_weight;

            // Skip if no input qubits but trying to add input faults
            if input_weight > 0 && self.config.input_qubits.is_empty() {
                continue;
            }

            // Enumerate input fault patterns of this weight
            for input_faults in self.enumerate_input_faults(input_weight) {
                // Enumerate internal fault patterns up to max_internal
                for internal_weight in 0..=max_internal {
                    self.analyze_internal_faults(
                        &input_faults,
                        input_weight,
                        internal_weight,
                        &mut analysis,
                        collect_failures,
                    );
                }
            }
        }

        analysis
    }

    /// Enumerate all input fault patterns of a given weight.
    fn enumerate_input_faults(&self, weight: usize) -> Vec<Vec<(usize, u8)>> {
        if weight == 0 {
            return vec![vec![]];
        }

        let pauli_types = self.config.pauli_types();
        if pauli_types.is_empty() {
            return vec![];
        }

        let qubits = &self.config.input_qubits;
        let mut results = Vec::new();

        // Generate all weight-w combinations of (qubit, pauli)
        for qubit_combo in combinations(qubits.len(), weight) {
            for pauli_combo in pauli_product(&pauli_types, weight) {
                let faults: Vec<(usize, u8)> = qubit_combo
                    .iter()
                    .zip(&pauli_combo)
                    .map(|(&q_idx, &p)| (qubits[q_idx], p))
                    .collect();
                results.push(faults);
            }
        }

        results
    }

    /// Analyze all internal fault patterns for a given input fault configuration.
    fn analyze_internal_faults(
        &self,
        input_faults: &[(usize, u8)],
        input_weight: usize,
        internal_weight: usize,
        analysis: &mut GadgetAnalysis,
        collect_failures: bool,
    ) {
        // Create iterator for internal faults
        let fault_config = FaultCheckConfig {
            max_weight: internal_weight,
            include_x: self.config.include_x,
            include_y: self.config.include_y,
            include_z: self.config.include_z,
            stop_on_first_failure: false,
            restricted_qubits: None,
            data_errors: true,
            ancilla_errors: true,
        };

        if internal_weight == 0 {
            // Only input faults, no internal faults
            let result = self.analyze_single_combination(
                input_faults,
                input_weight,
                &FaultConfiguration::new(),
                0,
            );
            self.record_result(result, analysis, collect_failures);
        } else {
            let fault_iter = PauliFaultIterator::new(
                self.internal_locations.clone(),
                internal_weight,
                fault_config,
            );

            for internal_config in fault_iter {
                let result = self.analyze_single_combination(
                    input_faults,
                    input_weight,
                    &internal_config,
                    internal_weight,
                );
                self.record_result(result, analysis, collect_failures);
            }
        }
    }

    /// Analyze a single (`input_faults`, `internal_faults`) combination.
    fn analyze_single_combination(
        &self,
        input_faults: &[(usize, u8)],
        input_weight: usize,
        internal_faults: &FaultConfiguration,
        internal_weight: usize,
    ) -> GadgetFaultResult {
        // Start with input faults
        let mut prop = PauliProp::new();
        for &(qubit, pauli) in input_faults {
            match pauli {
                1 => prop.add_x(qubit),
                2 => prop.add_y(qubit),
                3 => prop.add_z(qubit),
                _ => {}
            }
        }

        // Propagate through circuit with internal faults
        // We need to merge the initial prop with internal fault injection
        let prop = self.propagate_with_initial_error(prop, internal_faults);

        // Extract syndrome
        let z_syndrome_flips = self.get_syndrome_flips(&prop, &self.config.z_ancillas);
        let x_syndrome_flips = self.get_syndrome_flips(&prop, &self.config.x_ancillas);

        let has_syndrome = !z_syndrome_flips.is_empty() || !x_syndrome_flips.is_empty();

        // Check logical errors
        let logical_errors: Vec<bool> = self
            .config
            .logical_zs
            .iter()
            .chain(self.config.logical_xs.iter())
            .map(|(xs, zs)| self.anticommutes_with_logical(&prop, xs, zs))
            .collect();

        let has_logical_error = logical_errors.iter().any(|&e| e);

        // Calculate output error weight
        let output_error_weight = self.calculate_output_error_weight(&prop);

        // Classify
        let classification = self.classify(
            has_syndrome,
            has_logical_error,
            output_error_weight,
            input_weight + internal_weight,
        );

        GadgetFaultResult {
            input_weight,
            internal_weight,
            input_faults: input_faults.to_vec(),
            internal_faults: internal_faults.clone(),
            propagated_error: prop,
            z_syndrome_flips,
            x_syndrome_flips,
            logical_errors,
            output_error_weight,
            classification,
        }
    }

    /// Propagate through circuit with an initial error state and internal faults.
    fn propagate_with_initial_error(
        &self,
        initial: PauliProp,
        internal_faults: &FaultConfiguration,
    ) -> PauliProp {
        if internal_faults.is_empty() {
            // Just propagate the initial error through the circuit
            self.propagate_through_circuit(initial)
        } else {
            // Complex case: merge initial error propagation with fault injection
            // For now, we propagate initial and then apply internal faults
            // This is an approximation - proper handling requires interleaved injection
            let mut prop = self.propagate_through_circuit(initial);

            // Apply internal faults at the end (simplified)
            // TODO: Proper interleaved fault injection
            let internal_prop = propagate_faults(self.circuit, internal_faults);
            self.merge_pauli_props(&mut prop, &internal_prop);

            prop
        }
    }

    /// Propagate a `PauliProp` through the circuit without additional faults.
    fn propagate_through_circuit(&self, mut prop: PauliProp) -> PauliProp {
        for tick in self.circuit.ticks() {
            for gate in tick.gates() {
                let qubits: Vec<QubitId> = gate.qubits.to_vec();

                match gate.gate_type {
                    pecos_core::gate_type::GateType::H => {
                        prop.h(&qubits);
                    }
                    pecos_core::gate_type::GateType::SZ => {
                        prop.sz(&qubits);
                    }
                    pecos_core::gate_type::GateType::SZdg => {
                        prop.szdg(&qubits);
                    }
                    pecos_core::gate_type::GateType::CX => {
                        if qubits.len() >= 2 {
                            prop.cx(&[qubits[0], qubits[1]]);
                        }
                    }
                    pecos_core::gate_type::GateType::CZ => {
                        if qubits.len() >= 2 {
                            prop.cz(&[qubits[0], qubits[1]]);
                        }
                    }
                    pecos_core::gate_type::GateType::X => {
                        prop.x(&qubits);
                    }
                    pecos_core::gate_type::GateType::Y => {
                        prop.y(&qubits);
                    }
                    pecos_core::gate_type::GateType::Z => {
                        prop.z(&qubits);
                    }
                    // State prep and measurements don't affect Pauli propagation
                    _ => {}
                }
            }
        }

        prop
    }

    /// Merge two `PauliProp` states (XOR the X and Z components).
    fn merge_pauli_props(&self, target: &mut PauliProp, source: &PauliProp) {
        // XOR the X and Z bits from source into target
        for q in source.get_x_qubits() {
            target.add_x(q);
        }
        for q in source.get_z_qubits() {
            target.add_z(q);
        }
    }

    /// Get which syndrome qubits have flipped.
    fn get_syndrome_flips(&self, prop: &PauliProp, ancillas: &[usize]) -> Vec<usize> {
        ancillas
            .iter()
            .filter(|&&q| prop.contains_x(q))
            .copied()
            .collect()
    }

    /// Check if error anticommutes with a logical operator.
    fn anticommutes_with_logical(&self, prop: &PauliProp, xs: &[usize], zs: &[usize]) -> bool {
        // Error anticommutes with logical if odd number of anticommuting positions
        let mut count = 0;

        // X component of error anticommutes with Z component of logical
        for &q in &prop.get_x_qubits() {
            if zs.contains(&q) {
                count += 1;
            }
        }

        // Z component of error anticommutes with X component of logical
        for &q in &prop.get_z_qubits() {
            if xs.contains(&q) {
                count += 1;
            }
        }

        count % 2 == 1
    }

    /// Calculate the error weight on output qubits.
    fn calculate_output_error_weight(&self, prop: &PauliProp) -> usize {
        let output_set: BTreeSet<usize> = self.config.output_qubits.iter().copied().collect();

        let mut weight = 0;
        for &q in &prop.get_x_qubits() {
            if output_set.contains(&q) {
                weight += 1;
            }
        }
        for &q in &prop.get_z_qubits() {
            if output_set.contains(&q) && !prop.contains_x(q) {
                // Don't double count Y = XZ
                weight += 1;
            }
        }

        weight
    }

    /// Classify a fault combination result.
    fn classify(
        &self,
        has_syndrome: bool,
        has_logical_error: bool,
        output_error_weight: usize,
        total_fault_weight: usize,
    ) -> GadgetFaultClass {
        // For t-FT, output error weight should not exceed the fault weight
        // (errors shouldn't amplify beyond input)
        let excessive_output = output_error_weight > total_fault_weight;

        match (has_syndrome, has_logical_error, excessive_output) {
            (false, false, false) => GadgetFaultClass::Harmless,
            (true, false, false) => GadgetFaultClass::Correctable,
            (true, true, _) => GadgetFaultClass::DetectedLogicalError,
            (false, true, _) => GadgetFaultClass::UndetectedLogicalError,
            (_, _, true) => GadgetFaultClass::ExcessiveOutputError,
        }
    }

    /// Record a result in the analysis.
    fn record_result(
        &self,
        result: GadgetFaultResult,
        analysis: &mut GadgetAnalysis,
        collect_failures: bool,
    ) {
        analysis.total_tested += 1;

        match result.classification {
            GadgetFaultClass::Harmless => analysis.harmless += 1,
            GadgetFaultClass::Correctable => analysis.correctable += 1,
            GadgetFaultClass::DetectedLogicalError => analysis.detected_logical += 1,
            GadgetFaultClass::UndetectedLogicalError => {
                analysis.undetected_logical += 1;
                if collect_failures {
                    analysis.failures.push(result);
                }
            }
            GadgetFaultClass::ExcessiveOutputError => {
                analysis.excessive_output += 1;
                if collect_failures {
                    analysis.failures.push(result);
                }
            }
        }
    }

    // ========================================================================
    // Decoder Analysis Methods
    // ========================================================================

    /// Analyzes decoder requirements for this gadget.
    ///
    /// Groups faults by their syndrome pattern and identifies:
    /// - Correctable syndromes (all faults producing this syndrome are harmless)
    /// - Detected-uncorrectable syndromes (all faults cause logical error)
    /// - Ambiguous syndromes (some faults harmless, some cause logical error)
    ///
    /// A gadget is fault-tolerant from a decoder perspective if there are no
    /// undetectable logical errors and no ambiguous syndromes.
    #[must_use]
    pub fn analyze_decoder_requirements(&self, max_weight: usize) -> GadgetDecoderAnalysis {
        // Map from syndrome pattern to (correctable_count, uncorrectable_count)
        let mut syndrome_map: HashMap<Vec<u8>, (usize, usize)> = HashMap::new();
        let mut undetectable_logical_errors = 0;
        let mut undetectable_stabilizers = 0;
        let mut total_tested = 0;

        // Enumerate all (input_weight, internal_weight) combinations
        for input_weight in 0..=max_weight {
            let max_internal = max_weight - input_weight;

            if input_weight > 0 && self.config.input_qubits.is_empty() {
                continue;
            }

            for input_faults in self.enumerate_input_faults(input_weight) {
                for internal_weight in 0..=max_internal {
                    self.analyze_decoder_internal(
                        &input_faults,
                        input_weight,
                        internal_weight,
                        &mut syndrome_map,
                        &mut undetectable_logical_errors,
                        &mut undetectable_stabilizers,
                        &mut total_tested,
                    );
                }
            }
        }

        // Convert syndrome_map to analysis
        self.build_decoder_analysis(
            max_weight,
            total_tested,
            syndrome_map,
            undetectable_logical_errors,
            undetectable_stabilizers,
        )
    }

    /// Helper to analyze internal faults for decoder analysis.
    fn analyze_decoder_internal(
        &self,
        input_faults: &[(usize, u8)],
        input_weight: usize,
        internal_weight: usize,
        syndrome_map: &mut HashMap<Vec<u8>, (usize, usize)>,
        undetectable_logical_errors: &mut usize,
        undetectable_stabilizers: &mut usize,
        total_tested: &mut usize,
    ) {
        let fault_config = FaultCheckConfig {
            max_weight: internal_weight,
            include_x: self.config.include_x,
            include_y: self.config.include_y,
            include_z: self.config.include_z,
            stop_on_first_failure: false,
            restricted_qubits: None,
            data_errors: true,
            ancilla_errors: true,
        };

        if internal_weight == 0 {
            let result = self.analyze_single_combination(
                input_faults,
                input_weight,
                &FaultConfiguration::new(),
                0,
            );
            self.record_decoder_result(
                &result,
                syndrome_map,
                undetectable_logical_errors,
                undetectable_stabilizers,
            );
            *total_tested += 1;
        } else {
            let fault_iter = PauliFaultIterator::new(
                self.internal_locations.clone(),
                internal_weight,
                fault_config,
            );

            for internal_config in fault_iter {
                let result = self.analyze_single_combination(
                    input_faults,
                    input_weight,
                    &internal_config,
                    internal_weight,
                );
                self.record_decoder_result(
                    &result,
                    syndrome_map,
                    undetectable_logical_errors,
                    undetectable_stabilizers,
                );
                *total_tested += 1;
            }
        }
    }

    /// Record a fault result for decoder analysis.
    fn record_decoder_result(
        &self,
        result: &GadgetFaultResult,
        syndrome_map: &mut HashMap<Vec<u8>, (usize, usize)>,
        undetectable_logical_errors: &mut usize,
        undetectable_stabilizers: &mut usize,
    ) {
        // Build syndrome key
        let syndrome = self.build_syndrome_key(&result.z_syndrome_flips, &result.x_syndrome_flips);
        let has_logical_error = result.logical_errors.iter().any(|&e| e);

        // Check if syndrome is trivial (all zeros)
        let has_syndrome = syndrome.iter().any(|&s| s != 0);

        if has_syndrome {
            // Detectable - group by syndrome
            let entry = syndrome_map.entry(syndrome).or_insert((0, 0));
            if has_logical_error {
                entry.1 += 1; // uncorrectable
            } else {
                entry.0 += 1; // correctable
            }
        } else {
            // Undetectable
            if has_logical_error {
                *undetectable_logical_errors += 1;
            } else {
                *undetectable_stabilizers += 1;
            }
        }
    }

    /// Build a syndrome key from Z and X flips.
    fn build_syndrome_key(&self, z_flips: &[usize], x_flips: &[usize]) -> Vec<u8> {
        let mut syndrome = Vec::new();

        // Z ancillas (detect X errors)
        for &q in &self.config.z_ancillas {
            syndrome.push(u8::from(z_flips.contains(&q)));
        }

        // X ancillas (detect Z errors)
        for &q in &self.config.x_ancillas {
            syndrome.push(u8::from(x_flips.contains(&q)));
        }

        syndrome
    }

    /// Build decoder analysis from collected data.
    fn build_decoder_analysis(
        &self,
        max_weight: usize,
        total_tested: usize,
        syndrome_map: HashMap<Vec<u8>, (usize, usize)>,
        undetectable_logical_errors: usize,
        undetectable_stabilizers: usize,
    ) -> GadgetDecoderAnalysis {
        let mut syndromes = Vec::new();
        let mut correctable_syndromes = 0;
        let mut detected_uncorrectable_syndromes = 0;
        let mut ambiguous_syndromes = 0;

        for (syndrome, (correctable, uncorrectable)) in syndrome_map {
            let class = match (correctable > 0, uncorrectable > 0) {
                (true, false) => {
                    correctable_syndromes += 1;
                    SyndromeClass::Correctable
                }
                (false, true) => {
                    detected_uncorrectable_syndromes += 1;
                    SyndromeClass::DetectedUncorrectable
                }
                (true, true) => {
                    ambiguous_syndromes += 1;
                    SyndromeClass::Ambiguous
                }
                (false, false) => continue, // Empty - shouldn't happen
            };

            syndromes.push(GadgetSyndromeAnalysis {
                syndrome,
                correctable_count: correctable,
                uncorrectable_count: uncorrectable,
                class,
            });
        }

        GadgetDecoderAnalysis {
            max_weight,
            total_tested,
            syndromes,
            correctable_syndromes,
            detected_uncorrectable_syndromes,
            ambiguous_syndromes,
            undetectable_logical_errors,
            undetectable_stabilizers,
        }
    }

    // ========================================================================
    // Follow-Up Analysis Methods
    // ========================================================================

    /// Analyzes gadget with follow-up syndrome disambiguation.
    ///
    /// For gadgets with output qubits, the gadget's own syndrome may be ambiguous.
    /// By also considering what syndrome an ideal EC round following the gadget
    /// would produce, we can often disambiguate.
    ///
    /// The full syndrome key becomes: (`gadget_syndrome`, `follow_up_syndrome`)
    #[must_use]
    pub fn analyze_with_follow_up(
        &self,
        max_weight: usize,
        follow_up: &GadgetFollowUpConfig,
    ) -> GadgetDecoderAnalysis {
        let mut syndrome_map: HashMap<Vec<u8>, (usize, usize)> = HashMap::new();
        let mut undetectable_logical_errors = 0;
        let mut undetectable_stabilizers = 0;
        let mut total_tested = 0;

        for input_weight in 0..=max_weight {
            let max_internal = max_weight - input_weight;

            if input_weight > 0 && self.config.input_qubits.is_empty() {
                continue;
            }

            for input_faults in self.enumerate_input_faults(input_weight) {
                for internal_weight in 0..=max_internal {
                    self.analyze_follow_up_internal(
                        &input_faults,
                        input_weight,
                        internal_weight,
                        follow_up,
                        &mut syndrome_map,
                        &mut undetectable_logical_errors,
                        &mut undetectable_stabilizers,
                        &mut total_tested,
                    );
                }
            }
        }

        self.build_decoder_analysis(
            max_weight,
            total_tested,
            syndrome_map,
            undetectable_logical_errors,
            undetectable_stabilizers,
        )
    }

    /// Helper for follow-up analysis.
    #[allow(clippy::too_many_arguments)]
    fn analyze_follow_up_internal(
        &self,
        input_faults: &[(usize, u8)],
        input_weight: usize,
        internal_weight: usize,
        follow_up: &GadgetFollowUpConfig,
        syndrome_map: &mut HashMap<Vec<u8>, (usize, usize)>,
        undetectable_logical_errors: &mut usize,
        undetectable_stabilizers: &mut usize,
        total_tested: &mut usize,
    ) {
        let fault_config = FaultCheckConfig {
            max_weight: internal_weight,
            include_x: self.config.include_x,
            include_y: self.config.include_y,
            include_z: self.config.include_z,
            stop_on_first_failure: false,
            restricted_qubits: None,
            data_errors: true,
            ancilla_errors: true,
        };

        if internal_weight == 0 {
            let result = self.analyze_single_combination(
                input_faults,
                input_weight,
                &FaultConfiguration::new(),
                0,
            );
            self.record_follow_up_result(
                &result,
                follow_up,
                syndrome_map,
                undetectable_logical_errors,
                undetectable_stabilizers,
            );
            *total_tested += 1;
        } else {
            let fault_iter = PauliFaultIterator::new(
                self.internal_locations.clone(),
                internal_weight,
                fault_config,
            );

            for internal_config in fault_iter {
                let result = self.analyze_single_combination(
                    input_faults,
                    input_weight,
                    &internal_config,
                    internal_weight,
                );
                self.record_follow_up_result(
                    &result,
                    follow_up,
                    syndrome_map,
                    undetectable_logical_errors,
                    undetectable_stabilizers,
                );
                *total_tested += 1;
            }
        }
    }

    /// Record a result with follow-up syndrome.
    fn record_follow_up_result(
        &self,
        result: &GadgetFaultResult,
        follow_up: &GadgetFollowUpConfig,
        syndrome_map: &mut HashMap<Vec<u8>, (usize, usize)>,
        undetectable_logical_errors: &mut usize,
        undetectable_stabilizers: &mut usize,
    ) {
        // Build gadget syndrome
        let mut full_syndrome =
            self.build_syndrome_key(&result.z_syndrome_flips, &result.x_syndrome_flips);

        // Extract output error and compute follow-up syndrome
        let output_error =
            extract_output_error(&result.propagated_error, &self.config.output_qubits);

        // Convert stabilizers to the expected format
        let stabilizer_refs: Vec<(&[usize], &[usize])> = follow_up
            .follow_up_stabilizers
            .iter()
            .map(|(x, z)| (x.as_slice(), z.as_slice()))
            .collect();

        let follow_up_syndrome = compute_stabilizer_syndromes(&output_error, &stabilizer_refs);

        // Append follow-up syndrome to full syndrome
        for s in follow_up_syndrome {
            full_syndrome.push(u8::from(s));
        }

        let has_logical_error = result.logical_errors.iter().any(|&e| e);
        let has_syndrome = full_syndrome.iter().any(|&s| s != 0);

        if has_syndrome {
            let entry = syndrome_map.entry(full_syndrome).or_insert((0, 0));
            if has_logical_error {
                entry.1 += 1;
            } else {
                entry.0 += 1;
            }
        } else if has_logical_error {
            *undetectable_logical_errors += 1;
        } else {
            *undetectable_stabilizers += 1;
        }
    }

    // ========================================================================
    // Syndrome History Analysis Methods
    // ========================================================================

    /// Analyzes gadget considering full syndrome history across measurement rounds.
    ///
    /// For gadgets with multiple measurement rounds, this tracks the complete
    /// syndrome evolution rather than just the final syndrome. A fault may be
    /// detected in an earlier round even if the final syndrome is trivial.
    #[must_use]
    pub fn analyze_with_syndrome_history(&self, max_weight: usize) -> GadgetHistoryAnalysis {
        // Extract measurement rounds from circuit
        let rounds = extract_measurement_rounds(self.circuit);

        if rounds.is_empty() {
            // No measurement rounds - fall back to single-shot analysis
            let decoder_analysis = self.analyze_decoder_requirements(max_weight);
            return GadgetHistoryAnalysis {
                rounds: Vec::new(),
                histories: Vec::new(),
                correctable_histories: decoder_analysis.correctable_syndromes,
                uncorrectable_histories: decoder_analysis.detected_uncorrectable_syndromes,
                ambiguous_histories: decoder_analysis.ambiguous_syndromes,
                total_tested: decoder_analysis.total_tested,
                never_detected_logical_errors: decoder_analysis.undetectable_logical_errors,
                never_detected_stabilizers: decoder_analysis.undetectable_stabilizers,
            };
        }

        let mut history_map: HashMap<Vec<Vec<u8>>, (usize, usize)> = HashMap::new();
        let mut never_detected_logical_errors = 0;
        let mut never_detected_stabilizers = 0;
        let mut total_tested = 0;

        for input_weight in 0..=max_weight {
            let max_internal = max_weight - input_weight;

            if input_weight > 0 && self.config.input_qubits.is_empty() {
                continue;
            }

            for input_faults in self.enumerate_input_faults(input_weight) {
                for internal_weight in 0..=max_internal {
                    self.analyze_history_internal(
                        &input_faults,
                        input_weight,
                        internal_weight,
                        &rounds,
                        &mut history_map,
                        &mut never_detected_logical_errors,
                        &mut never_detected_stabilizers,
                        &mut total_tested,
                    );
                }
            }
        }

        self.build_history_analysis(
            rounds,
            total_tested,
            history_map,
            never_detected_logical_errors,
            never_detected_stabilizers,
        )
    }

    /// Helper for syndrome history analysis.
    #[allow(clippy::too_many_arguments)]
    fn analyze_history_internal(
        &self,
        input_faults: &[(usize, u8)],
        input_weight: usize,
        internal_weight: usize,
        rounds: &[MeasurementRound],
        history_map: &mut HashMap<Vec<Vec<u8>>, (usize, usize)>,
        never_detected_logical_errors: &mut usize,
        never_detected_stabilizers: &mut usize,
        total_tested: &mut usize,
    ) {
        let fault_config = FaultCheckConfig {
            max_weight: internal_weight,
            include_x: self.config.include_x,
            include_y: self.config.include_y,
            include_z: self.config.include_z,
            stop_on_first_failure: false,
            restricted_qubits: None,
            data_errors: true,
            ancilla_errors: true,
        };

        if internal_weight == 0 {
            self.record_history_result(
                input_faults,
                input_weight,
                &FaultConfiguration::new(),
                0,
                rounds,
                history_map,
                never_detected_logical_errors,
                never_detected_stabilizers,
            );
            *total_tested += 1;
        } else {
            let fault_iter = PauliFaultIterator::new(
                self.internal_locations.clone(),
                internal_weight,
                fault_config,
            );

            for internal_config in fault_iter {
                self.record_history_result(
                    input_faults,
                    input_weight,
                    &internal_config,
                    internal_weight,
                    rounds,
                    history_map,
                    never_detected_logical_errors,
                    never_detected_stabilizers,
                );
                *total_tested += 1;
            }
        }
    }

    /// Record a result with syndrome history.
    #[allow(clippy::too_many_arguments)]
    fn record_history_result(
        &self,
        input_faults: &[(usize, u8)],
        input_weight: usize,
        internal_faults: &FaultConfiguration,
        internal_weight: usize,
        rounds: &[MeasurementRound],
        history_map: &mut HashMap<Vec<Vec<u8>>, (usize, usize)>,
        never_detected_logical_errors: &mut usize,
        never_detected_stabilizers: &mut usize,
    ) {
        // Compute syndrome history
        let history =
            self.compute_syndrome_history(input_faults, internal_faults, internal_weight, rounds);

        // Check if any round had a syndrome
        let ever_detected = history
            .iter()
            .any(|round_syn| round_syn.iter().any(|&s| s != 0));

        // Check logical error (using final state)
        let result = self.analyze_single_combination(
            input_faults,
            input_weight,
            internal_faults,
            internal_weight,
        );
        let has_logical_error = result.logical_errors.iter().any(|&e| e);

        if ever_detected {
            let entry = history_map.entry(history).or_insert((0, 0));
            if has_logical_error {
                entry.1 += 1;
            } else {
                entry.0 += 1;
            }
        } else if has_logical_error {
            *never_detected_logical_errors += 1;
        } else {
            *never_detected_stabilizers += 1;
        }
    }

    /// Compute syndrome history for a fault configuration.
    fn compute_syndrome_history(
        &self,
        input_faults: &[(usize, u8)],
        internal_faults: &FaultConfiguration,
        internal_weight: usize,
        rounds: &[MeasurementRound],
    ) -> Vec<Vec<u8>> {
        let mut history = Vec::new();

        // Find the earliest tick where a fault occurs
        let earliest_fault_tick = if internal_weight > 0 {
            internal_faults
                .faults
                .iter()
                .map(|f| f.location.tick)
                .min()
                .unwrap_or(0)
        } else {
            0 // Input faults are at tick 0
        };

        for round in rounds {
            // Skip rounds before the fault could affect them
            if round.tick < earliest_fault_tick && input_faults.is_empty() {
                history.push(vec![
                    0;
                    self.config.z_ancillas.len()
                        + self.config.x_ancillas.len()
                ]);
                continue;
            }

            // Propagate up to this measurement round
            let prop = self.propagate_up_to_tick(input_faults, internal_faults, round.tick);

            // Build syndrome for this round
            let z_flips = self.get_syndrome_flips(&prop, &round.z_qubits);
            let x_flips = self.get_syndrome_flips(&prop, &round.x_qubits);

            let mut round_syndrome = Vec::new();
            for &q in &round.z_qubits {
                round_syndrome.push(u8::from(z_flips.contains(&q)));
            }
            for &q in &round.x_qubits {
                round_syndrome.push(u8::from(x_flips.contains(&q)));
            }

            history.push(round_syndrome);
        }

        history
    }

    /// Propagate faults up to a specific tick.
    fn propagate_up_to_tick(
        &self,
        input_faults: &[(usize, u8)],
        internal_faults: &FaultConfiguration,
        max_tick: usize,
    ) -> PauliProp {
        let mut prop = PauliProp::new();

        // Add input faults
        for &(qubit, pauli) in input_faults {
            match pauli {
                1 => prop.add_x(qubit),
                2 => prop.add_y(qubit),
                3 => prop.add_z(qubit),
                _ => {}
            }
        }

        // Propagate through circuit up to max_tick, injecting faults as we go
        for (tick_idx, tick) in self.circuit.ticks().iter().enumerate() {
            if tick_idx > max_tick {
                break;
            }

            // Inject faults before this tick
            for fault in &internal_faults.faults {
                if fault.location.tick == tick_idx && fault.location.before {
                    for (i, &p) in fault.paulis.iter().enumerate() {
                        if let Some(&qubit) = fault.location.qubits.get(i) {
                            match p {
                                1 => prop.add_x(qubit.0),
                                2 => prop.add_y(qubit.0),
                                3 => prop.add_z(qubit.0),
                                _ => {}
                            }
                        }
                    }
                }
            }

            // Apply gates
            for gate in tick.gates() {
                let qubits: Vec<QubitId> = gate.qubits.to_vec();
                match gate.gate_type {
                    pecos_core::gate_type::GateType::H => {
                        prop.h(&qubits);
                    }
                    pecos_core::gate_type::GateType::SZ => {
                        prop.sz(&qubits);
                    }
                    pecos_core::gate_type::GateType::SZdg => {
                        prop.szdg(&qubits);
                    }
                    pecos_core::gate_type::GateType::CX => {
                        if qubits.len() >= 2 {
                            prop.cx(&[qubits[0], qubits[1]]);
                        }
                    }
                    pecos_core::gate_type::GateType::CZ => {
                        if qubits.len() >= 2 {
                            prop.cz(&[qubits[0], qubits[1]]);
                        }
                    }
                    pecos_core::gate_type::GateType::X => {
                        prop.x(&qubits);
                    }
                    pecos_core::gate_type::GateType::Y => {
                        prop.y(&qubits);
                    }
                    pecos_core::gate_type::GateType::Z => {
                        prop.z(&qubits);
                    }
                    // State prep and measurements don't affect Pauli propagation
                    _ => {}
                }
            }

            // Inject faults after this tick
            for fault in &internal_faults.faults {
                if fault.location.tick == tick_idx && !fault.location.before {
                    for (i, &p) in fault.paulis.iter().enumerate() {
                        if let Some(&qubit) = fault.location.qubits.get(i) {
                            match p {
                                1 => prop.add_x(qubit.0),
                                2 => prop.add_y(qubit.0),
                                3 => prop.add_z(qubit.0),
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        prop
    }

    /// Build history analysis from collected data.
    fn build_history_analysis(
        &self,
        rounds: Vec<MeasurementRound>,
        total_tested: usize,
        history_map: HashMap<Vec<Vec<u8>>, (usize, usize)>,
        never_detected_logical_errors: usize,
        never_detected_stabilizers: usize,
    ) -> GadgetHistoryAnalysis {
        let mut histories = Vec::new();
        let mut correctable_histories = 0;
        let mut uncorrectable_histories = 0;
        let mut ambiguous_histories = 0;

        for (history, (correctable, uncorrectable)) in history_map {
            let class = match (correctable > 0, uncorrectable > 0) {
                (true, false) => {
                    correctable_histories += 1;
                    SyndromeClass::Correctable
                }
                (false, true) => {
                    uncorrectable_histories += 1;
                    SyndromeClass::DetectedUncorrectable
                }
                (true, true) => {
                    ambiguous_histories += 1;
                    SyndromeClass::Ambiguous
                }
                (false, false) => continue,
            };

            histories.push(GadgetHistoryPattern {
                history,
                correctable_count: correctable,
                uncorrectable_count: uncorrectable,
                class,
            });
        }

        GadgetHistoryAnalysis {
            rounds,
            histories,
            correctable_histories,
            uncorrectable_histories,
            ambiguous_histories,
            total_tested,
            never_detected_logical_errors,
            never_detected_stabilizers,
        }
    }
}

// ============================================================================
// Helper functions for enumeration
// ============================================================================

/// Generates all k-combinations of indices from 0..n.
fn combinations(n: usize, k: usize) -> impl Iterator<Item = Vec<usize>> {
    CombinationIterator::new(n, k)
}

struct CombinationIterator {
    n: usize,
    k: usize,
    indices: Vec<usize>,
    done: bool,
}

impl CombinationIterator {
    fn new(n: usize, k: usize) -> Self {
        if k > n || k == 0 {
            return Self {
                n,
                k,
                indices: Vec::new(),
                done: true,
            };
        }

        let indices: Vec<usize> = (0..k).collect();
        Self {
            n,
            k,
            indices,
            done: false,
        }
    }
}

impl Iterator for CombinationIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.indices.clone();

        // Advance to next combination
        let mut i = self.k;
        while i > 0 {
            i -= 1;
            if self.indices[i] < self.n - self.k + i {
                self.indices[i] += 1;
                for j in (i + 1)..self.k {
                    self.indices[j] = self.indices[j - 1] + 1;
                }
                return Some(result);
            }
        }

        self.done = true;
        Some(result)
    }
}

/// Generates all assignments of Pauli types to k positions.
fn pauli_product(types: &[u8], k: usize) -> impl Iterator<Item = Vec<u8>> + '_ {
    PauliProductIterator::new(types, k)
}

struct PauliProductIterator<'a> {
    types: &'a [u8],
    k: usize,
    indices: Vec<usize>,
    done: bool,
}

impl<'a> PauliProductIterator<'a> {
    fn new(types: &'a [u8], k: usize) -> Self {
        if types.is_empty() || k == 0 {
            return Self {
                types,
                k,
                indices: Vec::new(),
                done: true,
            };
        }

        Self {
            types,
            k,
            indices: vec![0; k],
            done: false,
        }
    }
}

impl Iterator for PauliProductIterator<'_> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result: Vec<u8> = self.indices.iter().map(|&i| self.types[i]).collect();

        // Advance to next assignment
        for i in (0..self.k).rev() {
            self.indices[i] += 1;
            if self.indices[i] < self.types.len() {
                return Some(result);
            }
            self.indices[i] = 0;
        }

        self.done = true;
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_syndrome_extraction_circuit() -> TickCircuit {
        let mut circuit = TickCircuit::new();
        // Don't initialize data qubits - they are INPUT
        circuit.tick().pz(&[3, 4]); // Initialize ancillas only
        circuit.tick().cx(&[(0, 3), (1, 4)]); // CNOT from data to ancilla
        circuit.tick().cx(&[(1, 3), (2, 4)]);
        circuit.tick().mz(&[3, 4]); // Measure ancillas
        // Data qubits are OUTPUT (not measured)
        circuit
    }

    fn build_state_prep_circuit() -> TickCircuit {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2]); // Initialize all data qubits
        circuit.tick().h(&[0]); // Some operations
        circuit.tick().cx(&[(0, 1), (0, 2)]);
        // Data qubits are OUTPUT (not measured)
        circuit
    }

    fn build_self_contained_circuit() -> TickCircuit {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2, 3, 4]);
        circuit.tick().cx(&[(0, 3), (1, 4)]);
        circuit.tick().cx(&[(1, 3), (2, 4)]);
        circuit.tick().mz(&[0, 1, 2, 3, 4]); // Measure everything
        circuit
    }

    #[test]
    fn test_gadget_config_builder() {
        let config = GadgetConfig::syndrome_extraction()
            .with_input_qubits(&[0, 1, 2])
            .with_output_qubits(&[0, 1, 2])
            .with_ancilla_qubits(&[3, 4])
            .with_z_ancillas(&[3, 4])
            .with_logical_z(&[], &[0, 1, 2]);

        assert_eq!(config.input_qubits, vec![0, 1, 2]);
        assert_eq!(config.output_qubits, vec![0, 1, 2]);
        assert!(config.has_input());
        assert!(config.has_output());
    }

    #[test]
    fn test_self_contained_gadget() {
        let circuit = build_self_contained_circuit();
        let config = GadgetConfig::self_contained()
            .with_z_ancillas(&[3, 4])
            .with_logical_z(&[], &[0, 1, 2]);

        let checker = GadgetChecker::new(&circuit, config);
        let analysis = checker.analyze(1);

        println!("Self-contained gadget analysis:");
        println!("  Total tested: {}", analysis.total_tested);
        println!("  Harmless: {}", analysis.harmless);
        println!("  Correctable: {}", analysis.correctable);
        println!("  Undetected logical: {}", analysis.undetected_logical);
    }

    #[test]
    fn test_syndrome_extraction_gadget() {
        let circuit = build_syndrome_extraction_circuit();
        let config = GadgetConfig::syndrome_extraction()
            .with_input_qubits(&[0, 1, 2])
            .with_output_qubits(&[0, 1, 2])
            .with_ancilla_qubits(&[3, 4])
            .with_z_ancillas(&[3, 4])
            .with_logical_z(&[], &[0, 1, 2]);

        let checker = GadgetChecker::new(&circuit, config);
        let analysis = checker.analyze(1);

        println!("Syndrome extraction gadget analysis:");
        println!("  Total tested: {}", analysis.total_tested);
        println!("  (includes input faults + internal faults with sum <= 1)");
        println!("  Harmless: {}", analysis.harmless);
        println!("  Correctable: {}", analysis.correctable);
        println!("  Undetected logical: {}", analysis.undetected_logical);
        println!("  Excessive output: {}", analysis.excessive_output);
    }

    #[test]
    fn test_state_prep_gadget() {
        let circuit = build_state_prep_circuit();
        let config = GadgetConfig::state_preparation()
            .with_output_qubits(&[0, 1, 2])
            .with_logical_z(&[], &[0, 1, 2]);

        let checker = GadgetChecker::new(&circuit, config);
        let analysis = checker.analyze(1);

        println!("State preparation gadget analysis:");
        println!("  Total tested: {}", analysis.total_tested);
        println!("  (no input faults, only internal faults)");
    }

    #[test]
    fn test_input_fault_enumeration() {
        let circuit = build_syndrome_extraction_circuit();
        let config = GadgetConfig::syndrome_extraction()
            .with_input_qubits(&[0, 1, 2])
            .with_output_qubits(&[0, 1, 2]);

        let checker = GadgetChecker::new(&circuit, config);

        // Weight-0 input faults: just empty
        let w0 = checker.enumerate_input_faults(0);
        assert_eq!(w0.len(), 1);
        assert!(w0[0].is_empty());

        // Weight-1 input faults: 3 qubits * 3 Pauli types = 9
        let w1 = checker.enumerate_input_faults(1);
        assert_eq!(w1.len(), 9);

        // Weight-2 input faults: C(3,2) * 3^2 = 3 * 9 = 27
        let w2 = checker.enumerate_input_faults(2);
        assert_eq!(w2.len(), 27);
    }

    #[test]
    fn test_gadget_fault_class() {
        assert!(GadgetFaultClass::UndetectedLogicalError.is_failure());
        assert!(GadgetFaultClass::ExcessiveOutputError.is_failure());
        assert!(!GadgetFaultClass::Harmless.is_failure());
        assert!(!GadgetFaultClass::Correctable.is_failure());
        assert!(!GadgetFaultClass::DetectedLogicalError.is_failure());
    }

    // ========================================================================
    // Tests for s + r <= t enumeration
    // ========================================================================

    #[test]
    fn test_s_plus_r_enumeration_t1() {
        // For t=1 with 3 input qubits:
        // - (s=0, r<=1): 1 * internal_faults_weight_0_and_1
        // - (s=1, r=0): 9 input patterns * 1 (no internal)
        let circuit = build_syndrome_extraction_circuit();
        let config = GadgetConfig::syndrome_extraction()
            .with_input_qubits(&[0, 1, 2])
            .with_output_qubits(&[0, 1, 2])
            .with_z_ancillas(&[3, 4]);

        let checker = GadgetChecker::new(&circuit, config);
        let analysis = checker.analyze(1);

        // We should have tested:
        // - (s=0, r=0): 1 case (no faults)
        // - (s=0, r=1): internal fault locations * 3 Pauli types
        // - (s=1, r=0): 9 input fault patterns
        // Total should include all these combinations
        assert!(
            analysis.total_tested > 0,
            "Should test at least some combinations"
        );

        // Verify we have input fault combinations by checking
        // that we tested more than just internal faults
        println!("t=1 analysis: {} total tested", analysis.total_tested);
    }

    #[test]
    fn test_s_plus_r_enumeration_t2() {
        // For t=2 with 3 input qubits, we should enumerate:
        // - (s=0, r<=2)
        // - (s=1, r<=1)
        // - (s=2, r=0)
        let circuit = build_syndrome_extraction_circuit();
        let config = GadgetConfig::syndrome_extraction()
            .with_input_qubits(&[0, 1, 2])
            .with_output_qubits(&[0, 1, 2])
            .with_z_ancillas(&[3, 4]);

        let checker = GadgetChecker::new(&circuit, config);
        let analysis_t2 = checker.analyze(2);
        let analysis_t1 = checker.analyze(1);

        // t=2 should test strictly more combinations than t=1
        assert!(
            analysis_t2.total_tested > analysis_t1.total_tested,
            "t=2 ({}) should test more than t=1 ({})",
            analysis_t2.total_tested,
            analysis_t1.total_tested
        );

        println!(
            "t=1: {} tested, t=2: {} tested",
            analysis_t1.total_tested, analysis_t2.total_tested
        );
    }

    #[test]
    fn test_input_only_vs_internal_only() {
        // Compare gadget with input qubits vs without
        let circuit = build_syndrome_extraction_circuit();

        // With input qubits
        let config_with_input = GadgetConfig::syndrome_extraction()
            .with_input_qubits(&[0, 1, 2])
            .with_output_qubits(&[0, 1, 2])
            .with_z_ancillas(&[3, 4]);

        // Without input qubits (self-contained style)
        let config_no_input = GadgetConfig::syndrome_extraction()
            .with_output_qubits(&[0, 1, 2])
            .with_z_ancillas(&[3, 4]);

        let checker_with = GadgetChecker::new(&circuit, config_with_input);
        let checker_without = GadgetChecker::new(&circuit, config_no_input);

        let analysis_with = checker_with.analyze(1);
        let analysis_without = checker_without.analyze(1);

        // With input qubits should test more combinations
        // (internal faults + input faults vs just internal faults)
        assert!(
            analysis_with.total_tested > analysis_without.total_tested,
            "With input qubits ({}) should test more than without ({})",
            analysis_with.total_tested,
            analysis_without.total_tested
        );

        println!(
            "With input: {} tested, Without input: {} tested",
            analysis_with.total_tested, analysis_without.total_tested
        );
    }

    #[test]
    fn test_specific_input_fault_counted() {
        // Verify that a specific input fault configuration is tested
        let circuit = build_syndrome_extraction_circuit();
        let config = GadgetConfig::syndrome_extraction()
            .with_input_qubits(&[0, 1, 2])
            .with_output_qubits(&[0, 1, 2])
            .with_z_ancillas(&[3, 4])
            .with_logical_z(&[], &[0, 1, 2]);

        let checker = GadgetChecker::new(&circuit, config);

        // Analyze with collecting failures
        let analysis = checker.analyze_with_options(1, true);

        // An X error on input qubit 0 should be detectable
        // (it will propagate to ancilla and flip syndrome)
        // So we shouldn't have undetected logical errors for weight-1

        // For this simple gadget, weight-1 input X errors on single qubits
        // should be correctable
        println!(
            "Weight-1 analysis: harmless={}, correctable={}, detected_logical={}, undetected_logical={}",
            analysis.harmless,
            analysis.correctable,
            analysis.detected_logical,
            analysis.undetected_logical
        );
    }

    #[test]
    fn test_combined_input_and_internal_fault() {
        // For t=2, we should test s=1 input + r=1 internal combinations
        let circuit = build_syndrome_extraction_circuit();
        let config = GadgetConfig::syndrome_extraction()
            .with_input_qubits(&[0, 1, 2])
            .with_output_qubits(&[0, 1, 2])
            .with_z_ancillas(&[3, 4])
            .with_logical_z(&[], &[0, 1, 2]);

        let checker = GadgetChecker::new(&circuit, config);

        // Analyze t=2
        let analysis = checker.analyze_with_options(2, true);

        // Count should include combined faults
        // The analysis should have run through (s=1, r=1) combinations
        println!(
            "t=2 combined faults: total={}, failures={}",
            analysis.total_tested,
            analysis.num_failures()
        );

        // For t=2, we should definitely have more than just weight-2 internal faults
        // or weight-2 input faults - we should also have weight-1 input + weight-1 internal
        let internal_only_count = {
            let config_no_input = GadgetConfig::syndrome_extraction()
                .with_output_qubits(&[0, 1, 2])
                .with_z_ancillas(&[3, 4])
                .with_logical_z(&[], &[0, 1, 2]);
            let checker = GadgetChecker::new(&circuit, config_no_input);
            checker.analyze(2).total_tested
        };

        assert!(
            analysis.total_tested > internal_only_count,
            "Combined input+internal ({}) should exceed internal-only ({})",
            analysis.total_tested,
            internal_only_count
        );
    }

    #[test]
    fn test_no_input_qubits_skips_input_faults() {
        // A gadget with no input qubits should not enumerate any input faults
        let circuit = build_state_prep_circuit();
        let config = GadgetConfig::state_preparation().with_output_qubits(&[0, 1, 2]);
        // Note: no input qubits specified

        let checker = GadgetChecker::new(&circuit, config);

        // Weight-1 input faults should return empty since no input qubits
        let input_faults_w1 = checker.enumerate_input_faults(1);
        assert!(
            input_faults_w1.is_empty(),
            "Should have no input faults when no input qubits"
        );

        // Only weight-0 (empty) should work
        let input_faults_w0 = checker.enumerate_input_faults(0);
        assert_eq!(input_faults_w0.len(), 1);
        assert!(input_faults_w0[0].is_empty());
    }

    #[test]
    fn test_fault_weight_breakdown() {
        // Explicitly verify the (s, r) breakdown for t=2
        let circuit = build_syndrome_extraction_circuit();
        let config = GadgetConfig::syndrome_extraction()
            .with_input_qubits(&[0, 1, 2]) // 3 input qubits
            .with_output_qubits(&[0, 1, 2])
            .with_z_ancillas(&[3, 4]);

        let checker = GadgetChecker::new(&circuit, config);

        // Count input fault patterns
        let input_w0 = checker.enumerate_input_faults(0).len(); // 1
        let input_w1 = checker.enumerate_input_faults(1).len(); // 9
        let input_w2 = checker.enumerate_input_faults(2).len(); // 27

        assert_eq!(input_w0, 1, "Weight-0 input should be 1 (empty)");
        assert_eq!(input_w1, 9, "Weight-1 input should be 3*3=9");
        assert_eq!(input_w2, 27, "Weight-2 input should be C(3,2)*9=27");

        // For t=2, the combinations are:
        // (s=0, r=0,1,2), (s=1, r=0,1), (s=2, r=0)
        // Total input patterns for these:
        // s=0: 1 pattern
        // s=1: 9 patterns
        // s=2: 27 patterns

        println!("Input fault patterns: w0={input_w0}, w1={input_w1}, w2={input_w2}");
    }

    #[test]
    fn test_gadget_config_from_circuit() {
        // Test auto-detection of I/O from circuit structure
        let circuit = build_syndrome_extraction_circuit();
        let config = GadgetConfig::from_circuit(&circuit);

        // Should detect data qubits 0, 1, 2 as inputs (used but not prepared)
        assert!(config.has_input(), "Should detect input qubits");
        assert_eq!(config.input_qubits.len(), 3);
        assert!(config.input_qubits.contains(&0));
        assert!(config.input_qubits.contains(&1));
        assert!(config.input_qubits.contains(&2));

        // Should detect data qubits 0, 1, 2 as outputs (used but not measured)
        assert!(config.has_output(), "Should detect output qubits");
        assert_eq!(config.output_qubits.len(), 3);

        // Should detect ancillas 3, 4 as ancilla qubits (prepared)
        assert_eq!(config.ancilla_qubits.len(), 2);
        assert!(config.ancilla_qubits.contains(&3));
        assert!(config.ancilla_qubits.contains(&4));
    }

    #[test]
    fn test_gadget_checker_from_circuit() {
        // Test the from_circuit constructor
        let circuit = build_syndrome_extraction_circuit();

        let checker = GadgetChecker::from_circuit(&circuit)
            .with_z_ancillas(&[3, 4])
            .with_logical_z(&[], &[0, 1, 2]);

        // Should have auto-detected I/O
        assert!(checker.has_input_qubits());
        assert!(checker.has_output_qubits());

        // Config should be set correctly
        assert_eq!(checker.config().z_ancillas, vec![3, 4]);
        assert_eq!(checker.config().logical_zs.len(), 1);

        // Accessor methods should return the correct values
        assert_eq!(checker.input_qubits().len(), 3);
        assert_eq!(checker.output_qubits().len(), 3);
    }

    #[test]
    fn test_gadget_config_with_detected_io() {
        // Test the with_detected_io method for auto-detecting from circuit
        let circuit = build_syndrome_extraction_circuit();

        let config = GadgetConfig::new()
            .with_detected_io(&circuit)
            .with_z_ancillas(&[3, 4]);

        // Should have auto-detected I/O
        assert_eq!(config.input_qubits.len(), 3);
        assert_eq!(config.output_qubits.len(), 3);
        assert_eq!(config.ancilla_qubits.len(), 2);
    }

    #[test]
    fn test_gadget_checker_state_prep_detection() {
        // State prep: all qubits prepared, no inputs
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2]); // All qubits prepared
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1), (0, 2)]);
        // No measurement - outputs go to next stage

        let checker = GadgetChecker::from_circuit(&circuit);

        assert!(!checker.has_input_qubits(), "State prep has no inputs");
        assert!(checker.has_output_qubits(), "State prep has outputs");

        assert!(checker.input_qubits().is_empty());
        assert_eq!(checker.output_qubits().len(), 3);
    }

    #[test]
    fn test_gadget_checker_final_measurement_detection() {
        // Final measurement: has inputs, no outputs
        let mut circuit = TickCircuit::new();
        // No prep - qubits come from previous stage
        circuit.tick().mz(&[0, 1, 2]); // Measure all

        let checker = GadgetChecker::from_circuit(&circuit).with_z_ancillas(&[0, 1, 2]);

        assert!(checker.has_input_qubits(), "Final measurement has inputs");
        assert!(
            !checker.has_output_qubits(),
            "Final measurement has no outputs"
        );

        assert_eq!(checker.input_qubits().len(), 3);
        assert!(checker.output_qubits().is_empty());
    }

    // =========================================================================
    // Tests for Decoder Analysis Methods
    // =========================================================================

    #[test]
    fn test_analyze_decoder_requirements() {
        // 3-qubit bit-flip syndrome extraction
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3), (1, 4)]);
        circuit.tick().cx(&[(1, 3), (2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let config = GadgetConfig::syndrome_extraction()
            .with_input_qubits(&[0, 1, 2])
            .with_output_qubits(&[0, 1, 2])
            .with_ancilla_qubits(&[3, 4])
            .with_z_ancillas(&[3, 4])
            .with_logical_z(&[], &[0, 1, 2]);

        let checker = GadgetChecker::new(&circuit, config);
        let analysis = checker.analyze_decoder_requirements(1);

        // Verify basic properties
        assert!(analysis.total_tested > 0);

        // The correctable + uncorrectable counts in syndromes should add up to
        // total_tested minus undetectable cases
        let syndrome_total: usize = analysis
            .syndromes
            .iter()
            .map(|s| s.correctable_count + s.uncorrectable_count)
            .sum();
        assert_eq!(
            syndrome_total
                + analysis.undetectable_logical_errors
                + analysis.undetectable_stabilizers,
            analysis.total_tested
        );

        // Syndrome category counts should match number of syndromes
        assert_eq!(
            analysis.correctable_syndromes
                + analysis.detected_uncorrectable_syndromes
                + analysis.ambiguous_syndromes,
            analysis.syndromes.len()
        );

        println!("Decoder requirements analysis:");
        println!("  Total tested: {}", analysis.total_tested);
        println!(
            "  Correctable syndromes: {}",
            analysis.correctable_syndromes
        );
        println!(
            "  Detected uncorrectable: {}",
            analysis.detected_uncorrectable_syndromes
        );
        println!("  Ambiguous syndromes: {}", analysis.ambiguous_syndromes);
        println!(
            "  Undetectable logical: {}",
            analysis.undetectable_logical_errors
        );
        println!(
            "  Undetectable stabilizer: {}",
            analysis.undetectable_stabilizers
        );
    }

    #[test]
    fn test_analyze_with_follow_up() {
        // 3-qubit bit-flip syndrome extraction
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3), (1, 4)]);
        circuit.tick().cx(&[(1, 3), (2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let config = GadgetConfig::syndrome_extraction()
            .with_input_qubits(&[0, 1, 2])
            .with_output_qubits(&[0, 1, 2])
            .with_ancilla_qubits(&[3, 4])
            .with_z_ancillas(&[3, 4])
            .with_logical_z(&[], &[0, 1, 2]);

        let checker = GadgetChecker::new(&circuit, config);

        // Follow-up stabilizers (same as the gadget would measure)
        let follow_up = GadgetFollowUpConfig::new(vec![
            (vec![], vec![0, 1]), // Z0Z1
            (vec![], vec![1, 2]), // Z1Z2
        ]);

        let analysis = checker.analyze_with_follow_up(1, &follow_up);

        // Verify basic properties
        assert!(analysis.total_tested > 0);

        // The syndrome vectors should now include follow-up syndrome bits
        for syndrome_analysis in &analysis.syndromes {
            // Original 2 ancillas + 2 follow-up stabilizers = 4 bits
            assert_eq!(syndrome_analysis.syndrome.len(), 4);
        }

        println!("Follow-up analysis:");
        println!("  Total tested: {}", analysis.total_tested);
        println!("  Is FT: {}", analysis.is_fault_tolerant());
        println!("  Syndrome patterns: {}", analysis.num_syndrome_patterns());
    }

    #[test]
    fn test_analyze_with_syndrome_history() {
        // Create a circuit with multiple measurement rounds
        let mut circuit = TickCircuit::new();

        // Round 1
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3), (1, 4)]);
        circuit.tick().cx(&[(1, 3), (2, 4)]);
        circuit.tick().mz(&[3, 4]);

        // Round 2 (repeat)
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3), (1, 4)]);
        circuit.tick().cx(&[(1, 3), (2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let config = GadgetConfig::syndrome_extraction()
            .with_input_qubits(&[0, 1, 2])
            .with_output_qubits(&[0, 1, 2])
            .with_ancilla_qubits(&[3, 4])
            .with_z_ancillas(&[3, 4])
            .with_logical_z(&[], &[0, 1, 2]);

        let checker = GadgetChecker::new(&circuit, config);
        let analysis = checker.analyze_with_syndrome_history(1);

        // Verify basic properties
        assert!(analysis.total_tested > 0);

        // Should have detected multiple rounds
        // (depending on how extract_measurement_rounds works)
        println!("Syndrome history analysis:");
        println!("  Rounds detected: {}", analysis.rounds.len());
        println!("  Total tested: {}", analysis.total_tested);
        println!(
            "  Correctable histories: {}",
            analysis.correctable_histories
        );
        println!(
            "  Uncorrectable histories: {}",
            analysis.uncorrectable_histories
        );
        println!("  Ambiguous histories: {}", analysis.ambiguous_histories);
        println!(
            "  Never detected logical: {}",
            analysis.never_detected_logical_errors
        );
        println!("  Is FT: {}", analysis.is_fault_tolerant());
    }

    #[test]
    fn test_gadget_decoder_analysis_methods() {
        // Quick test of the helper methods on GadgetDecoderAnalysis
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3), (1, 4)]);
        circuit.tick().cx(&[(1, 3), (2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let config = GadgetConfig::syndrome_extraction()
            .with_input_qubits(&[0, 1, 2])
            .with_output_qubits(&[0, 1, 2])
            .with_ancilla_qubits(&[3, 4])
            .with_z_ancillas(&[3, 4])
            .with_logical_z(&[], &[0, 1, 2]);

        let checker = GadgetChecker::new(&circuit, config);
        let analysis = checker.analyze_decoder_requirements(1);

        // Test helper methods
        let _ = analysis.is_fault_tolerant();
        let _ = analysis.num_syndrome_patterns();
        let problematic = analysis.problematic_syndromes();

        // Problematic syndromes should be non-correctable
        for syn in problematic {
            assert!(!matches!(syn.class, SyndromeClass::Correctable));
        }
    }
}
