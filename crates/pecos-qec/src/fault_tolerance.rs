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

//! Fault tolerance checking for quantum error correction.
//!
//! Provides stabilizer flip analysis, Pauli propagation, circuit-level checking,
//! and detector error model (DEM) generation.
//!
//! For a full guide, see `docs/user-guide/fault-tolerance.md`.

pub mod circuit_runner;
pub mod decoder_integration;
pub mod dem_builder;
pub mod gadget_checker;
pub mod influence_builder;
pub mod noisy_sampler;
pub mod pauli_prop_checker;
pub mod propagator;
pub mod stabilizer_flip_checker;

use pecos_core::QubitId;
use pecos_core::gate_type::GateType;
use std::collections::BTreeSet;

pub use circuit_runner::{
    FaultCategoryAnalysis, FaultChecker, extract_spacetime_locations, run_circuit_with_faults,
};
pub use decoder_integration::{
    CorrectionResult, ErrorCorrectionChecker, ErrorCorrectionConfig, ErrorCorrectionResult,
    LookupTableDecoder, apply_recovery, extract_syndrome, run_correction_cycle,
};
pub use gadget_checker::{
    GadgetAnalysis, GadgetChecker, GadgetConfig, GadgetDecoderAnalysis, GadgetFaultClass,
    GadgetFaultResult, GadgetFollowUpConfig, GadgetHistoryAnalysis, GadgetHistoryPattern,
    GadgetSyndromeAnalysis,
};
pub use influence_builder::InfluenceBuilder;
pub use pauli_prop_checker::{
    DecoderAnalysis, FaultClass, FaultToleranceAnalysis, FaultToleranceFailure, FollowUpConfig,
    MeasurementRound, PauliPropChecker, PropagationResult, SyndromeAnalysis, SyndromeClass,
    SyndromeHistory, SyndromeHistoryAnalysis, SyndromeHistoryResult, anticommutes_with_logical,
    classify_fault, compute_stabilizer_syndromes, detect_ancilla_qubits, detect_input_qubits,
    detect_output_qubits, extract_measurement_rounds, extract_output_error, get_syndrome_flips,
    has_syndrome, propagate_fault, propagate_faults,
};
pub use propagator::{
    DagFaultAnalyzer, DagFaultInfluenceMap, DagPropagator, DagSpacetimeLocation, DetectorId,
    Direction, FaultInfluence, FaultInfluenceMap, InfluenceBasedChecker, LogicalId, MeasurementId,
    TickFaultAnalyzer, apply_gate, propagate_backward_from_node, propagate_backward_from_tick,
    propagate_fault_backward, propagate_observable_backward, propagate_sparse_dag,
    propagate_through_circuit, propagate_through_dag, propagate_tick_range,
};
pub use stabilizer_flip_checker::{
    ErrorClass, StabilizerFlipAnalysis, StabilizerFlipChecker, StabilizerFlips,
};

/// A spacetime location where a fault can occur.
///
/// This represents a specific point in the circuit where an error
/// can be injected, including the timing (before or after the gate).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpacetimeLocation {
    /// The tick (time step) in the circuit.
    pub tick: usize,
    /// The qubit(s) involved in the gate at this location.
    pub qubits: Vec<QubitId>,
    /// Whether the error occurs before (true) or after (false) the gate.
    ///
    /// For measurements, errors typically occur before (affecting the syndrome).
    /// For other gates, errors typically occur after (representing gate errors).
    pub before: bool,
    /// The type of gate at this location.
    pub gate_type: GateType,
    /// Index of the gate within the tick (for circuits with multiple gates per tick).
    pub gate_index: usize,
}

impl SpacetimeLocation {
    /// Creates a new spacetime location.
    #[must_use]
    pub fn new(
        tick: usize,
        qubits: Vec<QubitId>,
        before: bool,
        gate_type: GateType,
        gate_index: usize,
    ) -> Self {
        Self {
            tick,
            qubits,
            before,
            gate_type,
            gate_index,
        }
    }

    /// Returns the number of qubits at this location.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Returns true if this is a measurement location.
    #[must_use]
    pub fn is_measurement(&self) -> bool {
        matches!(self.gate_type, GateType::MZ | GateType::MeasureFree)
    }
}

/// A Pauli error at a specific spacetime location.
///
/// The error is represented as a vector of single-qubit Paulis (I, X, Y, Z)
/// corresponding to each qubit at the location.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PauliFault {
    /// The spacetime location of this fault.
    pub location: SpacetimeLocation,
    /// The Pauli operator on each qubit (0=I, 1=X, 2=Y, 3=Z).
    pub paulis: Vec<u8>,
}

impl PauliFault {
    /// Creates a new Pauli fault.
    #[must_use]
    pub fn new(location: SpacetimeLocation, paulis: Vec<u8>) -> Self {
        Self { location, paulis }
    }

    /// Returns the weight (number of non-identity Paulis) of this fault.
    #[must_use]
    pub fn weight(&self) -> usize {
        self.paulis.iter().filter(|&&p| p != 0).count()
    }

    /// Returns true if this fault is non-trivial (not all identity).
    #[must_use]
    pub fn is_nontrivial(&self) -> bool {
        self.paulis.iter().any(|&p| p != 0)
    }

    /// Converts the Pauli indices to characters for display.
    #[must_use]
    pub fn pauli_string(&self) -> String {
        self.paulis
            .iter()
            .map(|&p| match p {
                0 => 'I',
                1 => 'X',
                2 => 'Y',
                3 => 'Z',
                _ => '?',
            })
            .collect()
    }
}

/// A collection of Pauli faults representing a fault configuration.
#[derive(Debug, Clone, Default)]
pub struct FaultConfiguration {
    /// The individual faults in this configuration.
    pub faults: Vec<PauliFault>,
}

impl FaultConfiguration {
    /// Creates a new empty fault configuration.
    #[must_use]
    pub fn new() -> Self {
        Self { faults: Vec::new() }
    }

    /// Creates a fault configuration with the given faults.
    #[must_use]
    pub fn with_faults(faults: Vec<PauliFault>) -> Self {
        Self { faults }
    }

    /// Adds a fault to this configuration.
    pub fn add_fault(&mut self, fault: PauliFault) {
        self.faults.push(fault);
    }

    /// Returns the total weight of all faults.
    #[must_use]
    pub fn total_weight(&self) -> usize {
        self.faults.iter().map(PauliFault::weight).sum()
    }

    /// Returns true if this configuration has no faults.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.faults.is_empty()
    }

    /// Returns the number of fault locations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.faults.len()
    }

    /// Groups faults by tick for error injection.
    ///
    /// Returns a map from tick index to (`before_faults`, `after_faults`).
    #[must_use]
    pub fn by_tick(
        &self,
    ) -> std::collections::BTreeMap<usize, (Vec<&PauliFault>, Vec<&PauliFault>)> {
        let mut result = std::collections::BTreeMap::new();
        for fault in &self.faults {
            let entry = result
                .entry(fault.location.tick)
                .or_insert_with(|| (Vec::new(), Vec::new()));
            if fault.location.before {
                entry.0.push(fault);
            } else {
                entry.1.push(fault);
            }
        }
        result
    }
}

/// Configuration for fault tolerance checking.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct FaultCheckConfig {
    /// Maximum weight of faults to check.
    pub max_weight: usize,
    /// Whether to include X-type errors.
    pub include_x: bool,
    /// Whether to include Y-type errors.
    pub include_y: bool,
    /// Whether to include Z-type errors.
    pub include_z: bool,
    /// Whether to stop at the first failure found.
    pub stop_on_first_failure: bool,
    /// Optional set of qubits to restrict fault locations to.
    pub restricted_qubits: Option<BTreeSet<QubitId>>,
    /// Whether to include errors on data qubits.
    pub data_errors: bool,
    /// Whether to include errors on ancilla qubits.
    pub ancilla_errors: bool,
}

impl Default for FaultCheckConfig {
    fn default() -> Self {
        Self {
            max_weight: 1,
            include_x: true,
            include_y: true,
            include_z: true,
            stop_on_first_failure: true,
            restricted_qubits: None,
            data_errors: true,
            ancilla_errors: false,
        }
    }
}

impl FaultCheckConfig {
    /// Creates a new configuration with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum weight of faults to check.
    #[must_use]
    pub fn with_weight(mut self, weight: usize) -> Self {
        self.max_weight = weight;
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

    /// Configures to check all Pauli types.
    #[must_use]
    pub fn all_paulis(mut self) -> Self {
        self.include_x = true;
        self.include_y = true;
        self.include_z = true;
        self
    }

    /// Sets whether to stop at the first failure.
    #[must_use]
    pub fn stop_on_first(mut self, stop: bool) -> Self {
        self.stop_on_first_failure = stop;
        self
    }

    /// Restricts fault locations to specific qubits.
    #[must_use]
    pub fn with_restricted_qubits(mut self, qubits: BTreeSet<QubitId>) -> Self {
        self.restricted_qubits = Some(qubits);
        self
    }

    /// Sets whether to include data qubit errors.
    #[must_use]
    pub fn with_data_errors(mut self, include: bool) -> Self {
        self.data_errors = include;
        self
    }

    /// Sets whether to include ancilla qubit errors.
    #[must_use]
    pub fn with_ancilla_errors(mut self, include: bool) -> Self {
        self.ancilla_errors = include;
        self
    }

    /// Returns the Pauli types to include based on configuration.
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
}

/// Iterator over all Pauli fault combinations of a given weight.
///
/// For a set of spacetime locations, this generates all ways to place
/// weight-w Pauli faults (excluding all-identity configurations).
pub struct PauliFaultIterator {
    /// The spacetime locations to place faults at.
    locations: Vec<SpacetimeLocation>,
    /// Configuration for which Pauli types to include.
    #[allow(dead_code)] // Stored for potential future filtering needs
    config: FaultCheckConfig,
    /// Current location combination (indices into `locations`).
    location_indices: Vec<usize>,
    /// Current Pauli combination for each location.
    pauli_indices: Vec<Vec<usize>>,
    /// Whether we've finished iterating.
    done: bool,
    /// Available Pauli types based on config.
    pauli_types: Vec<u8>,
}

impl PauliFaultIterator {
    /// Creates a new Pauli fault iterator.
    ///
    /// # Arguments
    ///
    /// * `locations` - All spacetime locations where faults can occur
    /// * `weight` - Number of fault locations to use
    /// * `config` - Configuration for Pauli types to include
    #[must_use]
    pub fn new(locations: Vec<SpacetimeLocation>, weight: usize, config: FaultCheckConfig) -> Self {
        let pauli_types = config.pauli_types();

        if weight == 0 || locations.is_empty() || pauli_types.is_empty() {
            return Self {
                locations,
                config,
                location_indices: Vec::new(),
                pauli_indices: Vec::new(),
                done: true,
                pauli_types,
            };
        }

        // Initialize with first `weight` locations
        let location_indices: Vec<usize> = (0..weight.min(locations.len())).collect();

        // Initialize Pauli indices for each location
        // For each qubit at each location, start with the first non-identity Pauli
        let pauli_indices: Vec<Vec<usize>> = location_indices
            .iter()
            .map(|&loc_idx| {
                let num_qubits = locations[loc_idx].num_qubits();
                vec![0; num_qubits] // Start with first Pauli type for each qubit
            })
            .collect();

        let done = weight > locations.len();

        Self {
            locations,
            config,
            location_indices,
            pauli_indices,
            done,
            pauli_types,
        }
    }

    /// Advances the Pauli combination for current locations.
    /// Returns true if successful, false if we need to advance locations.
    fn advance_paulis(&mut self) -> bool {
        // Treat pauli_indices as a mixed-radix number and increment
        for loc_idx in (0..self.pauli_indices.len()).rev() {
            let num_qubits = self.pauli_indices[loc_idx].len();
            for qubit_idx in (0..num_qubits).rev() {
                self.pauli_indices[loc_idx][qubit_idx] += 1;
                if self.pauli_indices[loc_idx][qubit_idx] < self.pauli_types.len() {
                    return true;
                }
                self.pauli_indices[loc_idx][qubit_idx] = 0;
            }
        }
        false
    }

    /// Advances to the next combination of locations.
    /// Returns true if successful, false if we've exhausted all combinations.
    fn advance_locations(&mut self) -> bool {
        let n = self.locations.len();
        let k = self.location_indices.len();

        // Find the rightmost index that can be incremented
        for i in (0..k).rev() {
            if self.location_indices[i] < n - k + i {
                self.location_indices[i] += 1;
                // Reset all indices to the right
                for j in (i + 1)..k {
                    self.location_indices[j] = self.location_indices[j - 1] + 1;
                }
                // Reset Pauli indices for new locations
                self.pauli_indices = self
                    .location_indices
                    .iter()
                    .map(|&loc_idx| {
                        let num_qubits = self.locations[loc_idx].num_qubits();
                        vec![0; num_qubits]
                    })
                    .collect();
                return true;
            }
        }
        false
    }

    /// Constructs the current fault configuration.
    fn current_configuration(&self) -> FaultConfiguration {
        let faults: Vec<PauliFault> = self
            .location_indices
            .iter()
            .zip(&self.pauli_indices)
            .map(|(&loc_idx, pauli_idx)| {
                let location = self.locations[loc_idx].clone();
                let paulis: Vec<u8> = pauli_idx.iter().map(|&idx| self.pauli_types[idx]).collect();
                PauliFault::new(location, paulis)
            })
            .collect();
        FaultConfiguration::with_faults(faults)
    }

    /// Checks if current configuration is non-trivial (not all identity on any location).
    fn is_nontrivial(&self) -> bool {
        // For weight > 0, we always have at least one non-I Pauli since we only
        // iterate over non-identity Paulis. But we should check that each
        // location has at least one non-identity Pauli.
        for pauli_idx in &self.pauli_indices {
            if pauli_idx.iter().all(|&idx| self.pauli_types[idx] == 0) {
                return false;
            }
        }
        true
    }
}

impl Iterator for PauliFaultIterator {
    type Item = FaultConfiguration;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.done {
            // Check if current configuration is valid
            if self.is_nontrivial() {
                let config = self.current_configuration();

                // Advance to next configuration
                if !self.advance_paulis() && !self.advance_locations() {
                    self.done = true;
                }

                return Some(config);
            }

            // Current is trivial, advance
            if !self.advance_paulis() && !self.advance_locations() {
                self.done = true;
            }
        }
        None
    }
}

/// Result of a fault tolerance check.
#[derive(Debug, Clone)]
pub struct FaultCheckResult {
    /// Fault configurations that caused failures.
    pub failures: Vec<FaultConfiguration>,
    /// Total number of configurations tested.
    pub total_tested: usize,
    /// Whether the circuit passed (no failures found).
    pub passed: bool,
    /// The weight that was tested.
    pub weight: usize,
}

impl FaultCheckResult {
    /// Creates a new fault check result.
    #[must_use]
    pub fn new(failures: Vec<FaultConfiguration>, total_tested: usize, weight: usize) -> Self {
        let passed = failures.is_empty();
        Self {
            failures,
            total_tested,
            passed,
            weight,
        }
    }

    /// Returns true if the circuit is fault-tolerant to the tested weight.
    #[must_use]
    pub fn is_fault_tolerant(&self) -> bool {
        self.passed
    }

    /// Returns the number of failures found.
    #[must_use]
    pub fn num_failures(&self) -> usize {
        self.failures.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spacetime_location() {
        let loc = SpacetimeLocation::new(0, vec![QubitId(0), QubitId(1)], false, GateType::CX, 0);
        assert_eq!(loc.tick, 0);
        assert_eq!(loc.num_qubits(), 2);
        assert!(!loc.is_measurement());
    }

    #[test]
    fn test_pauli_fault() {
        let loc = SpacetimeLocation::new(0, vec![QubitId(0)], false, GateType::H, 0);
        let fault = PauliFault::new(loc, vec![1]); // X error
        assert_eq!(fault.weight(), 1);
        assert!(fault.is_nontrivial());
        assert_eq!(fault.pauli_string(), "X");
    }

    #[test]
    fn test_pauli_fault_two_qubit() {
        let loc = SpacetimeLocation::new(0, vec![QubitId(0), QubitId(1)], false, GateType::CX, 0);
        let fault = PauliFault::new(loc, vec![1, 3]); // X on q0, Z on q1
        assert_eq!(fault.weight(), 2);
        assert_eq!(fault.pauli_string(), "XZ");
    }

    #[test]
    fn test_fault_configuration_by_tick() {
        let loc1 = SpacetimeLocation::new(0, vec![QubitId(0)], false, GateType::H, 0);
        let loc2 = SpacetimeLocation::new(1, vec![QubitId(0)], true, GateType::MZ, 0);

        let mut config = FaultConfiguration::new();
        config.add_fault(PauliFault::new(loc1, vec![1]));
        config.add_fault(PauliFault::new(loc2, vec![3]));

        let by_tick = config.by_tick();
        assert!(by_tick.contains_key(&0));
        assert!(by_tick.contains_key(&1));

        // Tick 0 has an after-fault
        assert_eq!(by_tick[&0].0.len(), 0); // before
        assert_eq!(by_tick[&0].1.len(), 1); // after

        // Tick 1 has a before-fault (measurement)
        assert_eq!(by_tick[&1].0.len(), 1); // before
        assert_eq!(by_tick[&1].1.len(), 0); // after
    }

    #[test]
    fn test_fault_check_config() {
        let config = FaultCheckConfig::new().with_weight(2).x_only();
        assert_eq!(config.max_weight, 2);
        assert!(config.include_x);
        assert!(!config.include_y);
        assert!(!config.include_z);
        assert_eq!(config.pauli_types(), vec![1]);
    }

    #[test]
    fn test_pauli_fault_iterator_single_location() {
        let locations = vec![SpacetimeLocation::new(
            0,
            vec![QubitId(0)],
            false,
            GateType::H,
            0,
        )];

        let config = FaultCheckConfig::new().all_paulis();
        let iter = PauliFaultIterator::new(locations, 1, config);

        let configs: Vec<_> = iter.collect();
        // Should generate X, Y, Z faults (3 total)
        assert_eq!(configs.len(), 3);
    }

    #[test]
    fn test_pauli_fault_iterator_weight_zero() {
        let locations = vec![SpacetimeLocation::new(
            0,
            vec![QubitId(0)],
            false,
            GateType::H,
            0,
        )];

        let config = FaultCheckConfig::new();
        let iter = PauliFaultIterator::new(locations, 0, config);

        let configs: Vec<_> = iter.collect();
        assert_eq!(configs.len(), 0);
    }

    #[test]
    fn test_pauli_fault_iterator_css_mode() {
        let locations = vec![SpacetimeLocation::new(
            0,
            vec![QubitId(0)],
            false,
            GateType::H,
            0,
        )];

        let config = FaultCheckConfig::new().z_only();
        let iter = PauliFaultIterator::new(locations, 1, config);

        let configs: Vec<_> = iter.collect();
        // Should generate only Z fault
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].faults[0].pauli_string(), "Z");
    }

    #[test]
    fn test_fault_check_result() {
        let result = FaultCheckResult::new(Vec::new(), 100, 1);
        assert!(result.is_fault_tolerant());
        assert_eq!(result.num_failures(), 0);
        assert_eq!(result.total_tested, 100);
    }
}
