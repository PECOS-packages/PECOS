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

//! Circuit runner with fault injection for fault tolerance checking.
//!
//! This module provides integration between the fault tolerance checking framework
//! and `TickCircuit` / `pecos-simulators` simulators.

use super::pauli_prop_checker::{CircuitIO, FaultClass, classify_fault, propagate_faults};
use super::{
    FaultCheckConfig, FaultCheckResult, FaultConfiguration, PauliFault, SpacetimeLocation,
};
use pecos_core::QubitId;
use pecos_core::gate_type::GateType;
use pecos_quantum::TickCircuit;
use pecos_simulators::CliffordGateable;

/// Extracts all spacetime locations from a `TickCircuit`.
///
/// This iterates through all gates in the circuit and creates spacetime
/// locations for fault injection.
///
/// # Arguments
///
/// * `circuit` - The tick circuit to analyze
/// * `include_initial` - Whether to include initial qubit locations (tick -1)
///
/// # Returns
///
/// A vector of all spacetime locations in the circuit.
#[must_use]
pub fn extract_spacetime_locations(
    circuit: &TickCircuit,
    include_initial: bool,
) -> Vec<SpacetimeLocation> {
    let mut locations = Vec::new();

    // Optionally include initial qubit locations (representing preparation errors)
    if include_initial {
        let all_qubits = circuit.all_qubits();
        for (idx, &qubit) in all_qubits.iter().enumerate() {
            locations.push(SpacetimeLocation::new(
                0, // Use tick 0 for initial, mark as "before" first gate
                vec![qubit],
                true, // Before any gates
                GateType::PZ,
                idx,
            ));
        }
    }

    // Iterate through all ticks
    for (tick_idx, tick) in circuit.iter_ticks() {
        for (gate_idx, gate) in tick.gates().iter().enumerate() {
            let qubits: Vec<QubitId> = gate.qubits.iter().copied().collect();
            let is_measurement = matches!(gate.gate_type, GateType::MZ | GateType::MeasureFree);

            locations.push(SpacetimeLocation::new(
                tick_idx,
                qubits,
                is_measurement, // Measurements get "before" errors
                gate.gate_type,
                gate_idx,
            ));
        }
    }

    locations
}

/// Applies a Pauli fault to a Clifford simulator.
///
/// # Arguments
///
/// * `sim` - The simulator to apply the fault to
/// * `fault` - The Pauli fault to apply
fn apply_fault<S: CliffordGateable>(sim: &mut S, fault: &PauliFault) {
    for (qubit, &pauli) in fault.location.qubits.iter().zip(&fault.paulis) {
        match pauli {
            1 => {
                sim.x(&[*qubit]);
            }
            2 => {
                sim.y(&[*qubit]);
            }
            3 => {
                sim.z(&[*qubit]);
            }
            _ => {} // Identity, do nothing
        }
    }
}

/// Applies a gate from a `TickCircuit` to a Clifford simulator.
///
/// # Arguments
///
/// * `sim` - The simulator to apply the gate to
/// * `gate` - The gate to apply
fn apply_gate<S: CliffordGateable>(sim: &mut S, gate: &pecos_core::Gate) {
    let qubits: Vec<QubitId> = gate.qubits.iter().copied().collect();

    match gate.gate_type {
        // Single-qubit gates
        GateType::I => {
            sim.identity(&qubits);
        }
        GateType::X => {
            sim.x(&qubits);
        }
        GateType::Y => {
            sim.y(&qubits);
        }
        GateType::Z => {
            sim.z(&qubits);
        }
        GateType::H => {
            sim.h(&qubits);
        }
        GateType::SX => {
            sim.sx(&qubits);
        }
        GateType::SXdg => {
            sim.sxdg(&qubits);
        }
        GateType::SY => {
            sim.sy(&qubits);
        }
        GateType::SYdg => {
            sim.sydg(&qubits);
        }
        GateType::SZ => {
            sim.sz(&qubits);
        }
        GateType::SZdg => {
            sim.szdg(&qubits);
        }

        // Two-qubit gates (qubits come in pairs: [ctrl, tgt, ctrl, tgt, ...])
        GateType::CX => {
            for pair in qubits.chunks(2) {
                if pair.len() == 2 {
                    sim.cx(&[pair[0], pair[1]]);
                }
            }
        }
        GateType::CY => {
            for pair in qubits.chunks(2) {
                if pair.len() == 2 {
                    sim.cy(&[pair[0], pair[1]]);
                }
            }
        }
        GateType::CZ => {
            for pair in qubits.chunks(2) {
                if pair.len() == 2 {
                    sim.cz(&[pair[0], pair[1]]);
                }
            }
        }
        GateType::SZZ => {
            for pair in qubits.chunks(2) {
                if pair.len() == 2 {
                    sim.szz(&[pair[0], pair[1]]);
                }
            }
        }
        GateType::SZZdg => {
            for pair in qubits.chunks(2) {
                if pair.len() == 2 {
                    sim.szzdg(&[pair[0], pair[1]]);
                }
            }
        }
        GateType::SWAP => {
            for pair in qubits.chunks(2) {
                if pair.len() == 2 {
                    sim.swap(&[pair[0], pair[1]]);
                }
            }
        }

        // Measurements
        GateType::MZ | GateType::MeasureFree => {
            sim.mz(&qubits);
        }

        // Preparations
        GateType::PZ => {
            sim.pz(&qubits);
        }

        // TODO: Add support for rotation gates if needed
        _ => {
            // Unsupported gate type - skip for now
        }
    }
}

/// Runs a circuit with fault injection on a Clifford simulator.
///
/// This executes the circuit tick by tick, injecting faults at the
/// specified spacetime locations.
///
/// # Arguments
///
/// * `circuit` - The circuit to run
/// * `sim` - The simulator to run on
/// * `faults` - The fault configuration to inject
///
/// # Returns
///
/// The simulator state after execution (for inspection).
pub fn run_circuit_with_faults<S: CliffordGateable>(
    circuit: &TickCircuit,
    sim: &mut S,
    faults: &FaultConfiguration,
) {
    // Group faults by tick
    let faults_by_tick = faults.by_tick();

    // Execute tick by tick
    for (tick_idx, tick) in circuit.iter_ticks() {
        // Get faults for this tick
        let empty: &[&PauliFault] = &[];
        let (before_faults, after_faults) = faults_by_tick
            .get(&tick_idx)
            .map_or((empty, empty), |(b, a)| (b.as_slice(), a.as_slice()));

        // Apply before-faults (typically for measurements)
        for fault in before_faults {
            apply_fault(sim, fault);
        }

        // Apply all gates in this tick
        for gate in tick.gates() {
            apply_gate(sim, gate);
        }

        // Apply after-faults (typical gate errors)
        for fault in after_faults {
            apply_fault(sim, fault);
        }
    }
}

/// Analysis of all fault categories in a circuit.
///
/// This categorizes every tested fault configuration into one of three
/// categories based on its effect on syndrome and logical operators.
#[derive(Debug, Clone)]
pub struct FaultCategoryAnalysis {
    /// Number of fault configurations that cause undetectable logical errors
    /// (no syndrome, but anticommutes with a logical operator).
    pub undetectable_logical_errors: usize,
    /// Number of fault configurations that cause undetectable stabilizer errors
    /// (no syndrome, commutes with all logical operators).
    pub undetectable_stabilizers: usize,
    /// Number of fault configurations that produce a syndrome
    /// (detectable, regardless of logical effect).
    pub detectable_errors: usize,
    /// Total number of fault configurations tested.
    pub total_tested: usize,
    /// Maximum fault weight tested.
    pub weight: usize,
    /// Detailed failure information (when `collect_failures` is true).
    pub failures: Vec<(FaultConfiguration, FaultClass)>,
}

impl FaultCategoryAnalysis {
    /// Returns true if there are no undetectable logical errors.
    #[must_use]
    pub fn is_fault_tolerant(&self) -> bool {
        self.undetectable_logical_errors == 0
    }

    /// Returns the fraction of faults that are detectable.
    #[must_use]
    pub fn detection_rate(&self) -> f64 {
        if self.total_tested == 0 {
            1.0
        } else {
            self.detectable_errors as f64 / self.total_tested as f64
        }
    }
}

/// A fault checker that tests a circuit for fault tolerance.
///
/// This provides a high-level API for checking whether a circuit is
/// fault-tolerant to weight-w Pauli faults.
///
/// # Example
///
/// ```
/// use pecos_qec::fault_tolerance::{FaultChecker, FaultCheckConfig};
/// use pecos_quantum::TickCircuit;
/// use pecos_simulators::SparseStab;
///
/// // Build a simple circuit
/// let mut circuit = TickCircuit::new();
/// circuit.tick().pz(&[0, 1, 2]);
/// circuit.tick().h(&[0]);
/// circuit.tick().cx(&[(0, 1)]);
/// circuit.tick().mz(&[0, 1]);
///
/// let checker = FaultChecker::new(&circuit)
///     .with_config(FaultCheckConfig::new().with_weight(1));
///
/// // Check all faults, returning true for any "failure"
/// // (here we just count - in practice you'd check logical errors)
/// let result = checker.check(
///     |_sim: &SparseStab| false,     // failure function
///     || SparseStab::new(3),         // simulator factory
/// );
/// assert!(result.is_fault_tolerant());
/// ```
pub struct FaultChecker<'a> {
    circuit: &'a TickCircuit,
    config: FaultCheckConfig,
    locations: Vec<SpacetimeLocation>,
    io: CircuitIO,
}

impl<'a> FaultChecker<'a> {
    /// Creates a new fault checker for the given circuit.
    #[must_use]
    pub fn new(circuit: &'a TickCircuit) -> Self {
        let locations = extract_spacetime_locations(circuit, false);
        let io = CircuitIO::from_circuit(circuit);
        Self {
            circuit,
            config: FaultCheckConfig::default(),
            locations,
            io,
        }
    }

    /// Sets the fault check configuration.
    #[must_use]
    pub fn with_config(mut self, config: FaultCheckConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets whether to include initial qubit locations.
    #[must_use]
    pub fn with_initial_locations(mut self, include: bool) -> Self {
        self.locations = extract_spacetime_locations(self.circuit, include);
        self
    }

    /// Returns the spacetime locations that will be checked.
    #[must_use]
    pub fn locations(&self) -> &[SpacetimeLocation] {
        &self.locations
    }

    /// Returns true if this circuit has input qubits.
    ///
    /// If true, fault tolerance analysis should consider s + r <= t enumeration
    /// to account for input faults.
    #[must_use]
    pub fn has_input_qubits(&self) -> bool {
        self.io.has_inputs()
    }

    /// Returns true if this circuit has output qubits.
    ///
    /// If true and analysis shows ambiguous syndromes, follow-up stabilizers
    /// may be needed to properly assess fault tolerance.
    #[must_use]
    pub fn has_output_qubits(&self) -> bool {
        self.io.has_outputs()
    }

    /// Returns the input qubits (used but never prepared).
    ///
    /// These qubits carry data/errors from a previous stage.
    #[must_use]
    pub fn input_qubits(&self) -> &[usize] {
        &self.io.input_qubits
    }

    /// Returns the output qubits (used but never measured).
    ///
    /// These qubits carry data/errors to the next stage.
    #[must_use]
    pub fn output_qubits(&self) -> &[usize] {
        &self.io.output_qubits
    }

    /// Returns the ancilla qubits (prepared within the circuit).
    #[must_use]
    pub fn ancilla_qubits(&self) -> &[usize] {
        &self.io.ancilla_qubits
    }

    /// Returns the measured qubits.
    #[must_use]
    pub fn measured_qubits(&self) -> &[usize] {
        &self.io.measured_qubits
    }

    /// Returns a description of the circuit type based on I/O structure.
    #[must_use]
    pub fn circuit_type(&self) -> &'static str {
        self.io.circuit_type()
    }

    /// Creates a fault iterator for the configured fault weight.
    fn create_fault_iterator(&self) -> super::PauliFaultIterator {
        super::PauliFaultIterator::new(
            self.locations.clone(),
            self.config.max_weight,
            self.config.clone(),
        )
    }

    /// Runs the fault tolerance check using the specified simulator type.
    ///
    /// # Arguments
    ///
    /// * `failure_fn` - A function that takes the simulator state after execution
    ///   and returns true if the result is a failure.
    /// * `sim_factory` - A function that creates a fresh simulator instance.
    ///
    /// # Returns
    ///
    /// The result of the fault tolerance check.
    pub fn check<S, F, Factory>(
        &self,
        mut failure_fn: F,
        mut sim_factory: Factory,
    ) -> FaultCheckResult
    where
        S: CliffordGateable,
        F: FnMut(&S) -> bool,
        Factory: FnMut() -> S,
    {
        let mut failures = Vec::new();
        let mut total_tested = 0;

        let fault_iter = self.create_fault_iterator();

        for fault_config in fault_iter {
            total_tested += 1;

            // Create fresh simulator
            let mut sim = sim_factory();

            // Run circuit with faults
            run_circuit_with_faults(self.circuit, &mut sim, &fault_config);

            // Check for failure
            if failure_fn(&sim) {
                failures.push(fault_config);

                if self.config.stop_on_first_failure {
                    break;
                }
            }
        }

        FaultCheckResult::new(failures, total_tested, self.config.max_weight)
    }

    /// Analyzes all fault configurations and categorizes them.
    ///
    /// This method uses Pauli propagation to efficiently classify each fault
    /// as either:
    /// - Undetectable logical error (no syndrome, affects logical)
    /// - Undetectable stabilizer (no syndrome, doesn't affect logical)
    /// - Detectable error (produces syndrome)
    ///
    /// # Arguments
    ///
    /// * `z_ancillas` - Qubits measured in Z basis (detect X errors)
    /// * `x_ancillas` - Qubits measured in X basis (detect Z errors)
    /// * `logicals` - Logical operators as (`x_positions`, `z_positions`) pairs
    /// * `collect_failures` - Whether to store detailed info for failures
    ///
    /// # Returns
    ///
    /// A `FaultCategoryAnalysis` with counts of each fault category.
    #[must_use]
    pub fn analyze_fault_categories(
        &self,
        z_ancillas: &[usize],
        x_ancillas: &[usize],
        logicals: &[(&[usize], &[usize])],
        collect_failures: bool,
    ) -> FaultCategoryAnalysis {
        let mut undetectable_logical_errors = 0;
        let mut undetectable_stabilizers = 0;
        let mut detectable_errors = 0;
        let mut total_tested = 0;
        let mut failures = Vec::new();

        let fault_iter = self.create_fault_iterator();

        for fault_config in fault_iter {
            total_tested += 1;

            // Use PauliProp to efficiently classify the fault
            let prop = propagate_faults(self.circuit, &fault_config);
            let classification = classify_fault(&prop, z_ancillas, x_ancillas, logicals);

            match classification {
                FaultClass::UndetectableLogicalError => {
                    undetectable_logical_errors += 1;
                    if collect_failures {
                        failures.push((fault_config, classification));
                    }
                }
                FaultClass::UndetectableStabilizer => {
                    undetectable_stabilizers += 1;
                }
                FaultClass::DetectableError => {
                    detectable_errors += 1;
                }
            }
        }

        FaultCategoryAnalysis {
            undetectable_logical_errors,
            undetectable_stabilizers,
            detectable_errors,
            total_tested,
            weight: self.config.max_weight,
            failures,
        }
    }

    /// Checks for undetectable logical errors (no syndrome, causes logical error).
    ///
    /// This is a convenience method that checks whether any fault configuration
    /// produces an undetectable logical error - the most critical failure mode.
    ///
    /// # Arguments
    ///
    /// * `z_ancillas` - Qubits measured in Z basis
    /// * `x_ancillas` - Qubits measured in X basis
    /// * `logicals` - Logical operators to check against
    ///
    /// # Returns
    ///
    /// A `FaultCheckResult` containing all fault configurations that cause
    /// undetectable logical errors.
    #[must_use]
    pub fn check_undetectable_logical_errors(
        &self,
        z_ancillas: &[usize],
        x_ancillas: &[usize],
        logicals: &[(&[usize], &[usize])],
    ) -> FaultCheckResult {
        let mut failures = Vec::new();
        let mut total_tested = 0;

        let fault_iter = self.create_fault_iterator();

        for fault_config in fault_iter {
            total_tested += 1;

            let prop = propagate_faults(self.circuit, &fault_config);
            let classification = classify_fault(&prop, z_ancillas, x_ancillas, logicals);

            if matches!(classification, FaultClass::UndetectableLogicalError) {
                failures.push(fault_config);

                if self.config.stop_on_first_failure {
                    break;
                }
            }
        }

        FaultCheckResult::new(failures, total_tested, self.config.max_weight)
    }

    /// Checks for any undetectable errors (no syndrome at all).
    ///
    /// This detects both undetectable logical errors and undetectable stabilizers.
    /// Useful for verifying syndrome extraction completeness.
    ///
    /// # Arguments
    ///
    /// * `z_ancillas` - Qubits measured in Z basis
    /// * `x_ancillas` - Qubits measured in X basis
    ///
    /// # Returns
    ///
    /// A `FaultCheckResult` containing all fault configurations that produce
    /// no syndrome (regardless of whether they cause logical errors).
    #[must_use]
    pub fn check_undetectable_errors(
        &self,
        z_ancillas: &[usize],
        x_ancillas: &[usize],
    ) -> FaultCheckResult {
        let mut failures = Vec::new();
        let mut total_tested = 0;

        let fault_iter = self.create_fault_iterator();

        // Empty logicals - we only care about syndrome detection
        let logicals: &[(&[usize], &[usize])] = &[];

        for fault_config in fault_iter {
            total_tested += 1;

            let prop = propagate_faults(self.circuit, &fault_config);
            let classification = classify_fault(&prop, z_ancillas, x_ancillas, logicals);

            // Both UndetectableLogicalError and UndetectableStabilizer have no syndrome
            if !matches!(classification, FaultClass::DetectableError) {
                failures.push(fault_config);

                if self.config.stop_on_first_failure {
                    break;
                }
            }
        }

        FaultCheckResult::new(failures, total_tested, self.config.max_weight)
    }

    /// Checks for output error weight expansion beyond a threshold.
    ///
    /// For gadgets with output qubits, this verifies that faults don't cause
    /// the output error weight to exceed a specified threshold.
    ///
    /// # Arguments
    ///
    /// * `output_qubits` - The qubits that carry output from this gadget
    /// * `max_output_weight` - Maximum acceptable output error weight
    ///
    /// # Returns
    ///
    /// A `FaultCheckResult` containing fault configurations where the output
    /// error weight exceeds the threshold.
    #[must_use]
    pub fn check_output_weight_expansion(
        &self,
        output_qubits: &[usize],
        max_output_weight: usize,
    ) -> FaultCheckResult {
        let mut failures = Vec::new();
        let mut total_tested = 0;

        let fault_iter = self.create_fault_iterator();

        for fault_config in fault_iter {
            total_tested += 1;

            let prop = propagate_faults(self.circuit, &fault_config);

            // Count non-identity Paulis on output qubits
            let output_weight: usize = output_qubits
                .iter()
                .filter(|&&q| prop.contains_x(q) || prop.contains_z(q))
                .count();

            if output_weight > max_output_weight {
                failures.push(fault_config);

                if self.config.stop_on_first_failure {
                    break;
                }
            }
        }

        FaultCheckResult::new(failures, total_tested, self.config.max_weight)
    }

    /// Runs the fault tolerance check with a full simulator while also tracking
    /// via Pauli propagation for syndrome/logical analysis.
    ///
    /// This allows combining full stabilizer simulation (for accurate final states)
    /// with efficient Pauli propagation (for syndrome/logical classification).
    ///
    /// # Arguments
    ///
    /// * `z_ancillas` - Qubits measured in Z basis
    /// * `x_ancillas` - Qubits measured in X basis
    /// * `logicals` - Logical operators to check
    /// * `sim_failure_fn` - Additional failure check using full simulator state
    /// * `sim_factory` - Factory to create simulator instances
    ///
    /// # Returns
    ///
    /// A `FaultCheckResult` containing faults that fail either the Pauli-based
    /// classification (undetectable logical error) or the simulator-based check.
    pub fn check_with_simulator<S, F, Factory>(
        &self,
        z_ancillas: &[usize],
        x_ancillas: &[usize],
        logicals: &[(&[usize], &[usize])],
        mut sim_failure_fn: F,
        mut sim_factory: Factory,
    ) -> FaultCheckResult
    where
        S: CliffordGateable,
        F: FnMut(&S, FaultClass) -> bool,
        Factory: FnMut() -> S,
    {
        let mut failures = Vec::new();
        let mut total_tested = 0;

        let fault_iter = self.create_fault_iterator();

        for fault_config in fault_iter {
            total_tested += 1;

            // Run both Pauli propagation and full simulation
            let prop = propagate_faults(self.circuit, &fault_config);
            let classification = classify_fault(&prop, z_ancillas, x_ancillas, logicals);

            // Run full simulator
            let mut sim = sim_factory();
            run_circuit_with_faults(self.circuit, &mut sim, &fault_config);

            // Check for failure using both sources of information
            if sim_failure_fn(&sim, classification) {
                failures.push(fault_config);

                if self.config.stop_on_first_failure {
                    break;
                }
            }
        }

        FaultCheckResult::new(failures, total_tested, self.config.max_weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_simulators::SparseStab;

    #[test]
    fn test_extract_spacetime_locations_empty_circuit() {
        let circuit = TickCircuit::new();
        let locations = extract_spacetime_locations(&circuit, false);
        assert!(locations.is_empty());
    }

    #[test]
    fn test_extract_spacetime_locations() {
        let mut circuit = TickCircuit::new();
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1)]);

        let locations = extract_spacetime_locations(&circuit, false);
        assert_eq!(locations.len(), 2);

        // First location: H gate at tick 0
        assert_eq!(locations[0].tick, 0);
        assert_eq!(locations[0].qubits, vec![QubitId(0)]);
        assert_eq!(locations[0].gate_type, GateType::H);

        // Second location: CX gate at tick 1
        assert_eq!(locations[1].tick, 1);
        assert_eq!(locations[1].qubits, vec![QubitId(0), QubitId(1)]);
        assert_eq!(locations[1].gate_type, GateType::CX);
    }

    #[test]
    fn test_fault_checker_creation() {
        let mut circuit = TickCircuit::new();
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1)]);

        let checker = FaultChecker::new(&circuit);
        assert_eq!(checker.locations().len(), 2);
    }

    // =========================================================================
    // Integration tests with real stabilizer codes
    // =========================================================================

    /// Build a simple Bell state preparation circuit.
    /// This is NOT fault-tolerant - a single X error on qubit 0 before the CX
    /// will propagate to both qubits.
    fn bell_state_circuit() -> TickCircuit {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1]); // Prepare |00>
        circuit.tick().h(&[0]); // Create |+0>
        circuit.tick().cx(&[(0, 1)]); // Create Bell state
        circuit
    }

    #[test]
    fn test_run_circuit_with_no_faults() {
        let circuit = bell_state_circuit();
        let mut sim = SparseStab::new(2);
        let empty_faults = FaultConfiguration::new();

        run_circuit_with_faults(&circuit, &mut sim, &empty_faults);

        // After Bell state prep, measuring both qubits should give correlated results
        // The state is (|00> + |11>)/sqrt(2)
        // We can't easily check the state, but we can verify no crash occurred
    }

    #[test]
    fn test_run_circuit_with_x_fault() {
        let circuit = bell_state_circuit();
        let mut sim = SparseStab::new(2);

        // Inject X error on qubit 0 after H gate
        let fault = PauliFault::new(
            SpacetimeLocation::new(1, vec![QubitId(0)], false, GateType::H, 0),
            vec![1], // X error
        );
        let faults = FaultConfiguration::with_faults(vec![fault]);

        run_circuit_with_faults(&circuit, &mut sim, &faults);
        // Circuit runs without crash - fault was injected
    }

    #[test]
    fn test_fault_checker_with_bell_state() {
        let circuit = bell_state_circuit();

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .all_paulis()
            .stop_on_first(false);

        let checker = FaultChecker::new(&circuit).with_config(config);

        // For Bell state, we define "failure" as never happening for this test
        // (we just want to verify the iteration works)
        let result = checker.check(|_sim: &SparseStab| false, || SparseStab::new(2));

        // Should have tested all weight-1 faults
        assert!(result.total_tested > 0);
        assert!(result.is_fault_tolerant()); // No failures since we always return false
    }

    /// Build a 3-qubit bit-flip code syndrome extraction circuit.
    ///
    /// Code: 3 data qubits (0, 1, 2), 2 ancilla qubits (3, 4)
    /// Stabilizers: Z0Z1 (ancilla 3), Z1Z2 (ancilla 4)
    /// Logical Z: Z0Z1Z2
    /// Logical X: X0X1X2
    fn three_qubit_bitflip_syndrome_circuit() -> TickCircuit {
        let mut circuit = TickCircuit::new();

        // Prepare ancillas in |0>
        circuit.tick().pz(&[3, 4]);

        // Measure Z0Z1 using ancilla 3
        circuit.tick().cx(&[(0, 3)]); // CNOT from data 0 to ancilla 3
        circuit.tick().cx(&[(1, 3)]); // CNOT from data 1 to ancilla 3

        // Measure Z1Z2 using ancilla 4
        circuit.tick().cx(&[(1, 4)]); // CNOT from data 1 to ancilla 4
        circuit.tick().cx(&[(2, 4)]); // CNOT from data 2 to ancilla 4

        // Measure ancillas
        circuit.tick().mz(&[3, 4]);

        circuit
    }

    #[test]
    fn test_three_qubit_code_syndrome_extraction() {
        let circuit = three_qubit_bitflip_syndrome_circuit();

        // Check we have the expected number of locations
        let locations = extract_spacetime_locations(&circuit, false);

        // 1 prep (2 qubits) + 4 CX gates + 1 measure (2 qubits) = 6 gate operations
        assert_eq!(locations.len(), 6);

        // Verify gate types
        assert_eq!(locations[0].gate_type, GateType::PZ);
        assert_eq!(locations[1].gate_type, GateType::CX);
        assert_eq!(locations[2].gate_type, GateType::CX);
        assert_eq!(locations[3].gate_type, GateType::CX);
        assert_eq!(locations[4].gate_type, GateType::CX);
        assert_eq!(locations[5].gate_type, GateType::MZ);
    }

    #[test]
    fn test_fault_checker_three_qubit_code() {
        let circuit = three_qubit_bitflip_syndrome_circuit();

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .all_paulis()
            .stop_on_first(false);

        let checker = FaultChecker::new(&circuit).with_config(config);

        // Count how many fault configurations we test
        let result = checker.check(|_sim: &SparseStab| false, || SparseStab::new(5));

        // We should test multiple configurations
        // 6 locations, each with X, Y, Z options
        // But some locations have multiple qubits (prep has 2, CX has 2, measure has 2)
        assert!(result.total_tested > 0);
        println!(
            "Tested {} fault configurations for 3-qubit code",
            result.total_tested
        );
    }

    #[test]
    fn test_fault_checker_css_mode() {
        let circuit = three_qubit_bitflip_syndrome_circuit();

        // Test X-only mode
        let config_x = FaultCheckConfig::new()
            .with_weight(1)
            .x_only()
            .stop_on_first(false);

        let checker_x = FaultChecker::new(&circuit).with_config(config_x);
        let result_x = checker_x.check(|_sim: &SparseStab| false, || SparseStab::new(5));

        // Test Z-only mode
        let config_z = FaultCheckConfig::new()
            .with_weight(1)
            .z_only()
            .stop_on_first(false);

        let checker_z = FaultChecker::new(&circuit).with_config(config_z);
        let result_z = checker_z.check(|_sim: &SparseStab| false, || SparseStab::new(5));

        // Test all-paulis mode
        let config_all = FaultCheckConfig::new()
            .with_weight(1)
            .all_paulis()
            .stop_on_first(false);

        let checker_all = FaultChecker::new(&circuit).with_config(config_all);
        let result_all = checker_all.check(|_sim: &SparseStab| false, || SparseStab::new(5));

        // X-only and Z-only should each test fewer configurations than all-paulis
        assert!(result_x.total_tested < result_all.total_tested);
        assert!(result_z.total_tested < result_all.total_tested);

        // X-only + Z-only should be less than all (since Y is excluded)
        assert!(result_x.total_tested + result_z.total_tested < result_all.total_tested);
    }

    #[test]
    fn test_fault_checker_weight_2() {
        let circuit = bell_state_circuit();

        let config = FaultCheckConfig::new()
            .with_weight(2)
            .all_paulis()
            .stop_on_first(false);

        let checker = FaultChecker::new(&circuit).with_config(config);
        let result = checker.check(|_sim: &SparseStab| false, || SparseStab::new(2));

        // Weight-2 should test more configurations than weight-1
        let config_w1 = FaultCheckConfig::new()
            .with_weight(1)
            .all_paulis()
            .stop_on_first(false);

        let checker_w1 = FaultChecker::new(&circuit).with_config(config_w1);
        let result_w1 = checker_w1.check(|_sim: &SparseStab| false, || SparseStab::new(2));

        assert!(result.total_tested > result_w1.total_tested);
    }

    /// Test that we can detect when a fault causes a measurable change.
    #[test]
    fn test_fault_detection_with_measurement() {
        // Simple circuit: prepare |0>, measure
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0]);
        circuit.tick().mz(&[0]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only() // X flips will change measurement outcome
            .stop_on_first(false);

        let checker = FaultChecker::new(&circuit).with_config(config);

        // Define failure as: measurement gave |1> instead of |0>
        // We detect this by checking if an X error was applied
        // (In a real test, we'd check the measurement outcome from the simulator)
        let mut x_faults_found = 0;
        let result = checker.check(
            |_sim: &SparseStab| {
                // In practice, we'd check sim's measurement results here
                // For now, just count
                x_faults_found += 1;
                false // Don't mark as failure
            },
            || SparseStab::new(1),
        );

        // Should have found X faults
        assert!(x_faults_found > 0);
        assert_eq!(x_faults_found, result.total_tested);
    }

    /// Test the Steane code syndrome extraction (7-qubit CSS code).
    fn steane_code_x_syndrome_circuit() -> TickCircuit {
        // Steane code: 7 data qubits (0-6), 3 X-syndrome ancillas (7, 8, 9)
        // X stabilizers check Z errors:
        // X0X1X2X3 (ancilla 7)
        // X0X1X4X5 (ancilla 8)
        // X0X2X4X6 (ancilla 9)

        let mut circuit = TickCircuit::new();

        // Prepare ancillas in |+> for X-type stabilizer measurement
        circuit.tick().pz(&[7, 8, 9]);
        circuit.tick().h(&[7, 8, 9]);

        // X0X1X2X3 measurement (ancilla 7)
        circuit.tick().cx(&[(7, 0)]);
        circuit.tick().cx(&[(7, 1)]);
        circuit.tick().cx(&[(7, 2)]);
        circuit.tick().cx(&[(7, 3)]);

        // X0X1X4X5 measurement (ancilla 8)
        circuit.tick().cx(&[(8, 0)]);
        circuit.tick().cx(&[(8, 1)]);
        circuit.tick().cx(&[(8, 4)]);
        circuit.tick().cx(&[(8, 5)]);

        // X0X2X4X6 measurement (ancilla 9)
        circuit.tick().cx(&[(9, 0)]);
        circuit.tick().cx(&[(9, 2)]);
        circuit.tick().cx(&[(9, 4)]);
        circuit.tick().cx(&[(9, 6)]);

        // Measure ancillas in X basis
        circuit.tick().h(&[7, 8, 9]);
        circuit.tick().mz(&[7, 8, 9]);

        circuit
    }

    #[test]
    fn test_steane_code_syndrome_extraction() {
        let circuit = steane_code_x_syndrome_circuit();

        let locations = extract_spacetime_locations(&circuit, false);

        // Count locations by type
        let preps = locations
            .iter()
            .filter(|l| l.gate_type == GateType::PZ)
            .count();
        let hadamards = locations
            .iter()
            .filter(|l| l.gate_type == GateType::H)
            .count();
        let cnots = locations
            .iter()
            .filter(|l| l.gate_type == GateType::CX)
            .count();
        let measures = locations
            .iter()
            .filter(|l| l.gate_type == GateType::MZ)
            .count();

        assert_eq!(preps, 1); // One bulk prep
        assert_eq!(hadamards, 2); // Two bulk H operations
        assert_eq!(cnots, 12); // 12 individual CX gates
        assert_eq!(measures, 1); // One bulk measure
    }

    #[test]
    fn test_steane_code_fault_enumeration() {
        let circuit = steane_code_x_syndrome_circuit();

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .z_only() // Only Z errors for X-stabilizer measurement
            .stop_on_first(false);

        let checker = FaultChecker::new(&circuit).with_config(config);
        let result = checker.check(|_sim: &SparseStab| false, || SparseStab::new(10));

        println!(
            "Steane code X-syndrome: tested {} weight-1 Z-fault configurations",
            result.total_tested
        );

        // Should have tested a reasonable number of configurations
        assert!(result.total_tested > 10);
    }

    #[test]
    fn test_fault_checker_circuit_io_detection() {
        // Syndrome extraction circuit: data qubits are inputs/outputs
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]); // Only ancillas prepared
        circuit.tick().cx(&[(0, 3), (1, 4)]);
        circuit.tick().cx(&[(1, 3), (2, 4)]);
        circuit.tick().mz(&[3, 4]); // Only ancillas measured

        let checker = FaultChecker::new(&circuit);

        // Should detect input and output qubits
        assert!(checker.has_input_qubits(), "Should detect input qubits");
        assert!(checker.has_output_qubits(), "Should detect output qubits");

        // Verify accessor methods
        assert_eq!(checker.input_qubits().len(), 3); // Data qubits 0, 1, 2
        assert_eq!(checker.output_qubits().len(), 3); // Data qubits 0, 1, 2
        assert_eq!(checker.ancilla_qubits().len(), 2); // Ancillas 3, 4
    }

    #[test]
    fn test_fault_checker_state_prep_io() {
        // State preparation: all qubits prepared, some measured
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2]); // All prepared
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1), (0, 2)]);
        // No measurement - outputs go to next stage

        let checker = FaultChecker::new(&circuit);

        assert!(!checker.has_input_qubits(), "State prep has no inputs");
        assert!(checker.has_output_qubits(), "State prep has outputs");

        assert!(checker.input_qubits().is_empty());
        assert_eq!(checker.output_qubits().len(), 3);
    }

    #[test]
    fn test_fault_checker_self_contained_io() {
        // Self-contained: all qubits prepared and measured
        let circuit = bell_state_circuit(); // Prep and no measurement

        let checker = FaultChecker::new(&circuit);

        // Bell state circuit preps both qubits but doesn't measure
        assert!(!checker.has_input_qubits(), "Self-contained has no inputs");
        assert!(checker.has_output_qubits(), "No measurement means outputs");
    }

    #[test]
    fn test_fault_checker_final_measurement_io() {
        // Final measurement: no prep, just measurement
        let mut circuit = TickCircuit::new();
        circuit.tick().mz(&[0, 1, 2]); // Measure all

        let checker = FaultChecker::new(&circuit);

        assert!(checker.has_input_qubits(), "Final measurement has inputs");
        assert!(
            !checker.has_output_qubits(),
            "Final measurement has no outputs"
        );

        assert_eq!(checker.input_qubits().len(), 3);
        assert!(checker.output_qubits().is_empty());
    }

    // =========================================================================
    // Real QEC Gadget CircuitIO Tests
    // =========================================================================

    #[test]
    fn test_circuit_io_three_qubit_code_syndrome_extraction() {
        // Real 3-qubit bit-flip code syndrome extraction
        // Data qubits 0,1,2 are inputs and outputs
        // Ancilla qubits 3,4 are prepared and measured
        let circuit = three_qubit_bitflip_syndrome_circuit();

        let checker = FaultChecker::new(&circuit);

        // Data qubits should be detected as inputs (used in CX but not prepared)
        assert!(checker.has_input_qubits());
        assert!(
            checker.input_qubits().contains(&0),
            "Data qubit 0 should be input"
        );
        assert!(
            checker.input_qubits().contains(&1),
            "Data qubit 1 should be input"
        );
        assert!(
            checker.input_qubits().contains(&2),
            "Data qubit 2 should be input"
        );
        assert!(
            !checker.input_qubits().contains(&3),
            "Ancilla 3 should not be input"
        );
        assert!(
            !checker.input_qubits().contains(&4),
            "Ancilla 4 should not be input"
        );

        // Data qubits should be detected as outputs (used but not measured)
        assert!(checker.has_output_qubits());
        assert!(
            checker.output_qubits().contains(&0),
            "Data qubit 0 should be output"
        );
        assert!(
            checker.output_qubits().contains(&1),
            "Data qubit 1 should be output"
        );
        assert!(
            checker.output_qubits().contains(&2),
            "Data qubit 2 should be output"
        );
        assert!(
            !checker.output_qubits().contains(&3),
            "Ancilla 3 should not be output"
        );
        assert!(
            !checker.output_qubits().contains(&4),
            "Ancilla 4 should not be output"
        );

        // Ancillas should be detected as prepared
        assert!(
            checker.ancilla_qubits().contains(&3),
            "Ancilla 3 should be prepared"
        );
        assert!(
            checker.ancilla_qubits().contains(&4),
            "Ancilla 4 should be prepared"
        );

        // Ancillas should be detected as measured
        assert!(
            checker.measured_qubits().contains(&3),
            "Ancilla 3 should be measured"
        );
        assert!(
            checker.measured_qubits().contains(&4),
            "Ancilla 4 should be measured"
        );

        println!("3-qubit code syndrome extraction:");
        println!("  Input qubits: {:?}", checker.input_qubits());
        println!("  Output qubits: {:?}", checker.output_qubits());
        println!("  Ancilla qubits: {:?}", checker.ancilla_qubits());
        println!("  Measured qubits: {:?}", checker.measured_qubits());
    }

    #[test]
    fn test_circuit_io_steane_code_x_syndrome() {
        // Real Steane code X-syndrome extraction
        // 7 data qubits (0-6), 3 ancilla qubits (7,8,9)
        let circuit = steane_code_x_syndrome_circuit();

        let checker = FaultChecker::new(&circuit);

        // All 7 data qubits should be inputs
        assert!(checker.has_input_qubits());
        for q in 0..7 {
            assert!(
                checker.input_qubits().contains(&q),
                "Data qubit {q} should be input"
            );
        }

        // All 7 data qubits should be outputs
        assert!(checker.has_output_qubits());
        for q in 0..7 {
            assert!(
                checker.output_qubits().contains(&q),
                "Data qubit {q} should be output"
            );
        }

        // Ancillas 7,8,9 should be prepared
        for q in 7..10 {
            assert!(
                checker.ancilla_qubits().contains(&q),
                "Ancilla {q} should be prepared"
            );
        }

        // Ancillas 7,8,9 should be measured
        for q in 7..10 {
            assert!(
                checker.measured_qubits().contains(&q),
                "Ancilla {q} should be measured"
            );
        }

        // Verify counts
        assert_eq!(
            checker.input_qubits().len(),
            7,
            "Should have 7 input qubits"
        );
        assert_eq!(
            checker.output_qubits().len(),
            7,
            "Should have 7 output qubits"
        );
        assert_eq!(
            checker.ancilla_qubits().len(),
            3,
            "Should have 3 ancilla qubits"
        );
        assert_eq!(
            checker.measured_qubits().len(),
            3,
            "Should have 3 measured qubits"
        );

        println!("Steane code X-syndrome extraction:");
        println!("  Input qubits: {:?}", checker.input_qubits());
        println!("  Output qubits: {:?}", checker.output_qubits());
        println!("  Ancilla qubits: {:?}", checker.ancilla_qubits());
    }

    /// Build a Steane code logical |0> state preparation circuit.
    fn steane_code_state_prep() -> TickCircuit {
        // Steane code |0_L> preparation using transversal encoding
        // All 7 qubits are prepared (no inputs)
        // All 7 qubits are outputs (no measurement)
        let mut circuit = TickCircuit::new();

        // Prepare all qubits in |0>
        circuit.tick().pz(&[0, 1, 2, 3, 4, 5, 6]);

        // Create superposition for encoding
        circuit.tick().h(&[0, 1, 3]);

        // Entangle to create logical |0>
        circuit.tick().cx(&[(0, 2), (1, 2)]);
        circuit.tick().cx(&[(0, 4), (1, 5), (3, 5)]);
        circuit.tick().cx(&[(0, 6), (1, 6), (3, 6)]);
        circuit.tick().cx(&[(3, 4)]);

        circuit
    }

    #[test]
    fn test_circuit_io_steane_code_state_prep() {
        // State preparation: all qubits prepared, none measured
        let circuit = steane_code_state_prep();

        let checker = FaultChecker::new(&circuit);

        // No input qubits (all are prepared within the gadget)
        assert!(
            !checker.has_input_qubits(),
            "State prep should have no inputs"
        );
        assert!(checker.input_qubits().is_empty());

        // All 7 qubits should be outputs
        assert!(
            checker.has_output_qubits(),
            "State prep should have outputs"
        );
        assert_eq!(
            checker.output_qubits().len(),
            7,
            "Should have 7 output qubits"
        );
        for q in 0..7 {
            assert!(
                checker.output_qubits().contains(&q),
                "Qubit {q} should be output"
            );
        }

        // All qubits are ancillas (prepared)
        assert_eq!(checker.ancilla_qubits().len(), 7);

        // No measured qubits
        assert!(checker.measured_qubits().is_empty());

        println!("Steane code state prep:");
        println!("  Circuit type: {}", checker.circuit_type());
        println!("  Output qubits: {:?}", checker.output_qubits());
    }

    /// Build a Steane code final measurement circuit.
    fn steane_code_final_measurement() -> TickCircuit {
        // Final destructive measurement of Steane code
        // All 7 data qubits are inputs (come from previous stage)
        // All 7 data qubits are measured (no outputs)
        let mut circuit = TickCircuit::new();

        // Just measure all data qubits
        circuit.tick().mz(&[0, 1, 2, 3, 4, 5, 6]);

        circuit
    }

    #[test]
    fn test_circuit_io_steane_code_final_measurement() {
        // Final measurement: all qubits are inputs, all are measured
        let circuit = steane_code_final_measurement();

        let checker = FaultChecker::new(&circuit);

        // All 7 qubits should be inputs (used but not prepared)
        assert!(
            checker.has_input_qubits(),
            "Final measurement should have inputs"
        );
        assert_eq!(
            checker.input_qubits().len(),
            7,
            "Should have 7 input qubits"
        );

        // No output qubits (all are measured)
        assert!(
            !checker.has_output_qubits(),
            "Final measurement should have no outputs"
        );
        assert!(checker.output_qubits().is_empty());

        // No ancillas (nothing is prepared)
        assert!(checker.ancilla_qubits().is_empty());

        // All qubits are measured
        assert_eq!(checker.measured_qubits().len(), 7);

        println!("Steane code final measurement:");
        println!("  Circuit type: {}", checker.circuit_type());
        println!("  Input qubits: {:?}", checker.input_qubits());
    }

    /// Build a complete single-round QEC circuit (state prep + syndrome + measurement).
    fn complete_three_qubit_qec() -> TickCircuit {
        let mut circuit = TickCircuit::new();

        // State preparation (prepare all data qubits)
        circuit.tick().pz(&[0, 1, 2]);

        // Encode into logical |0> (for 3-qubit code, this is just |000>)
        // Nothing to do for bit-flip code

        // Syndrome extraction
        circuit.tick().pz(&[3, 4]); // Prepare ancillas
        circuit.tick().cx(&[(0, 3), (1, 4)]);
        circuit.tick().cx(&[(1, 3), (2, 4)]);
        circuit.tick().mz(&[3, 4]); // Measure ancillas

        // Final measurement
        circuit.tick().mz(&[0, 1, 2]);

        circuit
    }

    #[test]
    fn test_circuit_io_complete_qec_round() {
        // Complete QEC: all qubits prepared, all measured (self-contained)
        let circuit = complete_three_qubit_qec();

        let checker = FaultChecker::new(&circuit);

        // No input qubits (all are prepared)
        assert!(
            !checker.has_input_qubits(),
            "Complete QEC should have no inputs"
        );
        assert!(checker.input_qubits().is_empty());

        // No output qubits (all are measured)
        assert!(
            !checker.has_output_qubits(),
            "Complete QEC should have no outputs"
        );
        assert!(checker.output_qubits().is_empty());

        // All 5 qubits should be ancillas (prepared)
        assert_eq!(checker.ancilla_qubits().len(), 5);

        // All 5 qubits should be measured
        assert_eq!(checker.measured_qubits().len(), 5);

        println!("Complete 3-qubit QEC round:");
        println!("  Circuit type: {}", checker.circuit_type());
        assert_eq!(
            checker.circuit_type(),
            "self-contained (state prep + final measurement)"
        );
    }

    /// Build a logical CNOT gadget between two 3-qubit codes.
    fn logical_cnot_gadget() -> TickCircuit {
        // Transversal CNOT between two 3-qubit codes
        // Control: qubits 0, 1, 2
        // Target: qubits 3, 4, 5
        // All 6 qubits are inputs and outputs (pass-through gadget)
        let mut circuit = TickCircuit::new();

        // Transversal CNOT: CX on each pair
        circuit.tick().cx(&[(0, 3), (1, 4), (2, 5)]);

        circuit
    }

    #[test]
    fn test_circuit_io_logical_cnot_gadget() {
        // Logical gate: all qubits are inputs and outputs
        let circuit = logical_cnot_gadget();

        let checker = FaultChecker::new(&circuit);

        // All 6 qubits should be inputs
        assert!(checker.has_input_qubits());
        assert_eq!(checker.input_qubits().len(), 6);

        // All 6 qubits should be outputs
        assert!(checker.has_output_qubits());
        assert_eq!(checker.output_qubits().len(), 6);

        // No ancillas or measurements
        assert!(checker.ancilla_qubits().is_empty());
        assert!(checker.measured_qubits().is_empty());

        println!("Logical CNOT gadget:");
        println!("  Circuit type: {}", checker.circuit_type());
        assert_eq!(
            checker.circuit_type(),
            "pass-through gadget (has inputs and outputs)"
        );
    }

    /// Build a flag-based syndrome extraction circuit.
    fn flagged_syndrome_extraction() -> TickCircuit {
        // Simplified flag-based syndrome extraction
        // Data qubits: 0, 1, 2
        // Syndrome ancilla: 3
        // Flag ancilla: 4
        let mut circuit = TickCircuit::new();

        // Prepare ancillas
        circuit.tick().pz(&[3, 4]);

        // First part of weight-2 stabilizer with flag
        circuit.tick().cx(&[(3, 4)]); // Entangle syndrome with flag
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(3, 4)]); // Disentangle

        // Measure both ancillas
        circuit.tick().mz(&[3, 4]);

        circuit
    }

    #[test]
    fn test_circuit_io_flagged_syndrome() {
        // Flag-based syndrome extraction
        let circuit = flagged_syndrome_extraction();

        let checker = FaultChecker::new(&circuit);

        // Data qubits 0, 1 should be inputs (used in CX but not prepared)
        // Note: qubit 2 is not used in this simplified circuit
        assert!(checker.has_input_qubits());
        assert!(checker.input_qubits().contains(&0));
        assert!(checker.input_qubits().contains(&1));

        // Data qubits should be outputs (not measured)
        assert!(checker.has_output_qubits());
        assert!(checker.output_qubits().contains(&0));
        assert!(checker.output_qubits().contains(&1));

        // Both ancillas should be prepared and measured
        assert!(checker.ancilla_qubits().contains(&3));
        assert!(checker.ancilla_qubits().contains(&4));
        assert!(checker.measured_qubits().contains(&3));
        assert!(checker.measured_qubits().contains(&4));

        println!("Flagged syndrome extraction:");
        println!("  Input qubits: {:?}", checker.input_qubits());
        println!("  Output qubits: {:?}", checker.output_qubits());
        println!("  Ancilla qubits: {:?}", checker.ancilla_qubits());
    }

    // =========================================================================
    // Tests for Built-in Failure Detectors
    // =========================================================================

    #[test]
    fn test_analyze_fault_categories() {
        // Use the three-qubit bit-flip syndrome extraction circuit
        let circuit = three_qubit_bitflip_syndrome_circuit();

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .all_paulis()
            .stop_on_first(false);

        let checker = FaultChecker::new(&circuit).with_config(config);

        // Z ancillas are 3, 4 (detecting X errors)
        let z_ancillas = &[3, 4];
        let x_ancillas: &[usize] = &[];
        // Logical Z operator for 3-qubit code is Z0Z1Z2
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];

        let analysis = checker.analyze_fault_categories(z_ancillas, x_ancillas, logicals, false);

        // Verify basic properties
        assert!(analysis.total_tested > 0);
        assert_eq!(
            analysis.total_tested,
            analysis.undetectable_logical_errors
                + analysis.undetectable_stabilizers
                + analysis.detectable_errors
        );

        println!("Fault category analysis:");
        println!("  Total tested: {}", analysis.total_tested);
        println!(
            "  Undetectable logical errors: {}",
            analysis.undetectable_logical_errors
        );
        println!(
            "  Undetectable stabilizers: {}",
            analysis.undetectable_stabilizers
        );
        println!("  Detectable errors: {}", analysis.detectable_errors);
        println!(
            "  Detection rate: {:.2}%",
            analysis.detection_rate() * 100.0
        );
    }

    #[test]
    fn test_check_undetectable_logical_errors() {
        // Use the three-qubit bit-flip syndrome extraction circuit
        let circuit = three_qubit_bitflip_syndrome_circuit();

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .all_paulis()
            .stop_on_first(false);

        let checker = FaultChecker::new(&circuit).with_config(config);

        let z_ancillas = &[3, 4];
        let x_ancillas: &[usize] = &[];
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];

        let result = checker.check_undetectable_logical_errors(z_ancillas, x_ancillas, logicals);

        assert!(result.total_tested > 0);

        // For single weight-1 faults, the 3-qubit code should detect all X errors
        // but may have issues with Z errors that affect logical Z
        println!(
            "Undetectable logical errors check: {} failures out of {} tested",
            result.failures.len(),
            result.total_tested
        );
    }

    #[test]
    fn test_check_undetectable_errors() {
        let circuit = three_qubit_bitflip_syndrome_circuit();

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .all_paulis()
            .stop_on_first(false);

        let checker = FaultChecker::new(&circuit).with_config(config);

        let z_ancillas = &[3, 4];
        let x_ancillas: &[usize] = &[];

        let result = checker.check_undetectable_errors(z_ancillas, x_ancillas);

        assert!(result.total_tested > 0);

        println!(
            "Undetectable errors check: {} failures out of {} tested",
            result.failures.len(),
            result.total_tested
        );
    }

    #[test]
    fn test_check_output_weight_expansion() {
        let circuit = three_qubit_bitflip_syndrome_circuit();

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .all_paulis()
            .stop_on_first(false);

        let checker = FaultChecker::new(&circuit).with_config(config);

        // Output qubits are data qubits 0, 1, 2
        let output_qubits = &[0, 1, 2];

        // Check that weight-1 faults don't expand to weight > 1 on outputs
        let result = checker.check_output_weight_expansion(output_qubits, 1);

        assert!(result.total_tested > 0);

        println!(
            "Output weight expansion check: {} failures out of {} tested (weight expansion > 1)",
            result.failures.len(),
            result.total_tested
        );
    }

    #[test]
    fn test_analyze_fault_categories_with_collection() {
        let circuit = three_qubit_bitflip_syndrome_circuit();

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .all_paulis()
            .stop_on_first(false);

        let checker = FaultChecker::new(&circuit).with_config(config);

        let z_ancillas = &[3, 4];
        let x_ancillas: &[usize] = &[];
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];

        // Collect failure details
        let analysis = checker.analyze_fault_categories(z_ancillas, x_ancillas, logicals, true);

        // Verify that failures are collected
        assert_eq!(
            analysis.failures.len(),
            analysis.undetectable_logical_errors
        );

        // Verify each collected failure has the right classification
        for (fault_config, classification) in &analysis.failures {
            assert!(
                matches!(classification, FaultClass::UndetectableLogicalError),
                "Collected failure should be UndetectableLogicalError"
            );
            assert!(
                !fault_config.faults.is_empty(),
                "Fault config should have at least one fault"
            );
        }

        println!("Collected {} failure details", analysis.failures.len());
    }

    #[test]
    fn test_fault_category_analysis_methods() {
        let circuit = three_qubit_bitflip_syndrome_circuit();

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .all_paulis()
            .stop_on_first(false);

        let checker = FaultChecker::new(&circuit).with_config(config);

        let z_ancillas = &[3, 4];
        let x_ancillas: &[usize] = &[];
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];

        let analysis = checker.analyze_fault_categories(z_ancillas, x_ancillas, logicals, false);

        // Test is_fault_tolerant method
        let expected_ft = analysis.undetectable_logical_errors == 0;
        assert_eq!(analysis.is_fault_tolerant(), expected_ft);

        // Test detection_rate method
        let expected_rate = analysis.detectable_errors as f64 / analysis.total_tested as f64;
        assert!((analysis.detection_rate() - expected_rate).abs() < 1e-10);

        println!(
            "Is fault tolerant: {}, Detection rate: {:.2}%",
            analysis.is_fault_tolerant(),
            analysis.detection_rate() * 100.0
        );
    }

    #[test]
    fn test_check_with_simulator() {
        // Test check_with_simulator which combines Pauli propagation with full simulation
        let circuit = three_qubit_bitflip_syndrome_circuit();

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .all_paulis()
            .stop_on_first(false);

        let checker = FaultChecker::new(&circuit).with_config(config);

        let z_ancillas = &[3, 4];
        let x_ancillas: &[usize] = &[];
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];

        // Use check_with_simulator with a failure function that checks for
        // undetectable logical errors (using the FaultClass from Pauli propagation)
        let result = checker.check_with_simulator(
            z_ancillas,
            x_ancillas,
            logicals,
            |_sim: &SparseStab, classification| {
                // Fail if we have an undetectable logical error
                matches!(classification, FaultClass::UndetectableLogicalError)
            },
            || SparseStab::new(5),
        );

        assert!(result.total_tested > 0);

        // The result should match what we get from check_undetectable_logical_errors
        let direct_result =
            checker.check_undetectable_logical_errors(z_ancillas, x_ancillas, logicals);

        assert_eq!(
            result.failures.len(),
            direct_result.failures.len(),
            "check_with_simulator should find same failures as check_undetectable_logical_errors"
        );

        println!(
            "check_with_simulator: {} failures out of {} tested",
            result.failures.len(),
            result.total_tested
        );
    }
}
