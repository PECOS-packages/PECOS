//! Influence Map Builder
//!
//! Complete pipeline for building fault influence maps from circuits:
//!
//! 1. **Forward symbolic simulation** (`SymbolicSparseStab`) to determine measurement correlations
//! 2. **Detector definition** from deterministic measurements
//! 3. **Backward propagation** to build the influence map
//!
//! # Example
//!
//! ```ignore
//! use pecos_qec::fault_tolerance::InfluenceBuilder;
//! use pecos_quantum::DagCircuit;
//!
//! // Build a syndrome extraction circuit
//! let mut dag = DagCircuit::new();
//! // ... add gates ...
//!
//! // Build influence map with automatic detector discovery
//! let builder = InfluenceBuilder::new(&dag);
//! let influence_map = builder.build();
//!
//! // Use with CPU sampler
//! let mut sampler = NoisySampler::new(&influence_map, noise_model, seed);
//! let results = sampler.sample(10000);
//! ```

use super::propagator::dag::{DagFaultInfluenceMap, DagSpacetimeLocation};
use super::propagator::types::{DetectorId, MeasurementId};
use super::propagator::{DagFaultAnalyzer, DagPropagator, Direction, Pauli, apply_gate};
use pecos_core::QubitId;
use pecos_qsim::{PauliProp, SymbolicSparseStab};
use smallvec::SmallVec;
use std::collections::BinaryHeap;

/// Builder for fault influence maps with proper detector definitions.
///
/// This integrates forward symbolic simulation with backward propagation
/// to create complete influence maps suitable for noisy sampling.
pub struct InfluenceBuilder<'a> {
    dag: &'a pecos_quantum::DagCircuit,
    /// Logical operators to track (qubit indices with X or Z component)
    logical_x_qubits: Vec<usize>,
    logical_z_qubits: Vec<usize>,
}

impl<'a> InfluenceBuilder<'a> {
    /// Create a new influence builder for the given circuit.
    #[must_use]
    pub fn new(dag: &'a pecos_quantum::DagCircuit) -> Self {
        Self {
            dag,
            logical_x_qubits: Vec::new(),
            logical_z_qubits: Vec::new(),
        }
    }

    /// Add a logical X operator to track.
    ///
    /// The logical X is defined as X on all specified qubits.
    #[must_use]
    pub fn with_logical_x(mut self, qubits: Vec<usize>) -> Self {
        self.logical_x_qubits = qubits;
        self
    }

    /// Add a logical Z operator to track.
    ///
    /// The logical Z is defined as Z on all specified qubits.
    #[must_use]
    pub fn with_logical_z(mut self, qubits: Vec<usize>) -> Self {
        self.logical_z_qubits = qubits;
        self
    }

    /// Build the influence map.
    ///
    /// This performs:
    /// 1. Forward symbolic simulation to get measurement correlations
    /// 2. Detector extraction from deterministic measurements
    /// 3. Backward propagation from detectors and logicals
    #[must_use]
    pub fn build(&self) -> DagFaultInfluenceMap {
        // Step 1: Run forward symbolic simulation
        let measurement_info = self.run_symbolic_simulation();

        // Step 2: Extract detectors from deterministic measurements
        let detectors = self.extract_detectors(&measurement_info);

        // Step 3: Build influence map with backward propagation
        self.build_influence_map_with_detectors(&measurement_info, &detectors)
    }

    /// Run symbolic simulation to get measurement correlations.
    fn run_symbolic_simulation(&self) -> MeasurementInfo {
        // Determine number of qubits from the circuit
        let max_qubit = self
            .dag
            .topological_order()
            .iter()
            .filter_map(|&node| self.dag.gate(node))
            .flat_map(|op| op.qubits.iter())
            .map(pecos_core::QubitId::index)
            .max()
            .unwrap_or(0);

        let num_qubits = max_qubit + 1;
        let mut sim = SymbolicSparseStab::new(num_qubits);

        // Track node -> measurement index mapping
        let mut node_to_meas_idx: Vec<Option<usize>> = vec![None; self.dag.gate_count() + 1];
        let mut meas_idx = 0;

        // Execute circuit symbolically
        for &node in &self.dag.topological_order() {
            if let Some(op) = self.dag.gate(node) {
                let qubits: Vec<usize> = op.qubits.iter().map(pecos_core::QubitId::index).collect();

                match op.gate_type {
                    pecos_quantum::GateType::H => {
                        sim.h(qubits[0]);
                    }
                    pecos_quantum::GateType::SZ => {
                        sim.sz(qubits[0]);
                    }
                    pecos_quantum::GateType::SZdg => {
                        // SZdg = SZ^3 = SZ * SZ * SZ
                        sim.sz(qubits[0]);
                        sim.sz(qubits[0]);
                        sim.sz(qubits[0]);
                    }
                    pecos_quantum::GateType::X => {
                        sim.x(qubits[0]);
                    }
                    pecos_quantum::GateType::Y => {
                        sim.y(qubits[0]);
                    }
                    pecos_quantum::GateType::Z => {
                        sim.z(qubits[0]);
                    }
                    pecos_quantum::GateType::CX => {
                        sim.cx(qubits[0], qubits[1]);
                    }
                    pecos_quantum::GateType::CZ => {
                        // CZ = H(target) CX H(target)
                        sim.h(qubits[1]);
                        sim.cx(qubits[0], qubits[1]);
                        sim.h(qubits[1]);
                    }
                    pecos_quantum::GateType::MZ | pecos_quantum::GateType::MeasureFree => {
                        sim.mz(qubits[0]);
                        node_to_meas_idx[node] = Some(meas_idx);
                        meas_idx += 1;
                    }
                    // Skip other gates (identity, barriers, Prep, etc.)
                    _ => {}
                }
            }
        }

        let history = sim.measurement_history().clone();

        MeasurementInfo {
            history,
            node_to_meas_idx,
            num_measurements: meas_idx,
        }
    }

    /// Extract detectors from deterministic measurements.
    ///
    /// A deterministic measurement `m_i` with outcome `{m_a, m_b, ...}` means
    /// that in the noiseless case: `m_i = m_a XOR m_b XOR ...`
    ///
    /// This defines a detector: `D = m_i XOR m_a XOR m_b XOR ... = 0` always.
    fn extract_detectors(&self, info: &MeasurementInfo) -> Vec<DetectorDef> {
        let mut detectors = Vec::new();

        for result in info.history.iter() {
            if result.is_deterministic {
                // This measurement is deterministic - it defines a detector
                let mut measurement_indices: SmallVec<[usize; 4]> = SmallVec::new();

                // Add the measurement itself
                measurement_indices.push(result.index);

                // Add all its dependencies (the XOR terms)
                for dep_idx in &result.outcome {
                    measurement_indices.push(dep_idx);
                }

                // Account for flip (if flip=true, the detector should be 1, not 0)
                // For now, we treat flipped detectors the same way
                detectors.push(DetectorDef {
                    measurement_indices,
                    expected_value: result.flip,
                });
            }
        }

        detectors
    }

    /// Build influence map with proper detector definitions.
    fn build_influence_map_with_detectors(
        &self,
        info: &MeasurementInfo,
        detectors: &[DetectorDef],
    ) -> DagFaultInfluenceMap {
        let analyzer = DagFaultAnalyzer::new(self.dag);
        let propagator = analyzer.propagator();

        let num_locations = analyzer.propagator().topo_order().len() * 2; // rough estimate
        let mut map = DagFaultInfluenceMap::with_capacity(num_locations);

        // Copy locations from analyzer
        map.locations = self.extract_locations(propagator);

        // Build measurement node lookup
        let measurements = self.extract_measurements(propagator);
        map.measurements.clone_from(&measurements);

        // Create DetectorId entries for each detector
        for detector in detectors {
            let meas_ids: SmallVec<[MeasurementId; 2]> = detector
                .measurement_indices
                .iter()
                .filter_map(|&meas_idx| {
                    // Find the node for this measurement index
                    info.node_to_meas_idx
                        .iter()
                        .position(|&opt| opt == Some(meas_idx))
                        .map(|node| {
                            // Find qubit for this measurement
                            let qubit = measurements
                                .iter()
                                .find(|&&(n, _, _)| n == node)
                                .map_or(0, |&(_, q, _)| q);
                            MeasurementId {
                                tick: node,
                                qubit,
                                basis: 0, // Z-basis
                            }
                        })
                })
                .collect();

            map.detectors.push(DetectorId {
                measurements: meas_ids,
                name: None,
            });
        }

        // Add logical operators as additional "detectors" for tracking
        let num_detectors = detectors.len();
        let mut num_logicals = 0;

        // Track logical X (sensitive to Z errors)
        if !self.logical_x_qubits.is_empty() {
            num_logicals += 1;
        }
        // Track logical Z (sensitive to X errors)
        if !self.logical_z_qubits.is_empty() {
            num_logicals += 1;
        }

        // Build the influence structure using backward propagation
        let mut recorder = CompoundRecorder::new(map.locations.len(), num_detectors, num_logicals);

        // Propagate from each detector
        self.propagate_detectors(propagator, info, detectors, &mut recorder);

        // Propagate from logicals
        self.propagate_logicals(propagator, &mut recorder);

        // Convert to SoA format
        map.influences = recorder.into_soa();

        map
    }

    /// Extract fault locations from the propagator.
    fn extract_locations(&self, propagator: &DagPropagator<'_>) -> Vec<DagSpacetimeLocation> {
        let mut locations = Vec::new();

        for &node in propagator.topo_order() {
            if let Some(gate) = propagator.gate(node) {
                let qubits: Vec<QubitId> = gate.qubits.to_vec();

                let is_measurement = matches!(
                    gate.gate_type,
                    pecos_quantum::GateType::MZ | pecos_quantum::GateType::MeasureFree
                );
                let is_prep = matches!(
                    gate.gate_type,
                    pecos_quantum::GateType::PZ | pecos_quantum::GateType::QAlloc
                );

                if is_measurement {
                    locations.push(DagSpacetimeLocation {
                        node,
                        qubits: qubits.clone(),
                        before: true,
                        gate_type: gate.gate_type,
                    });
                } else if is_prep {
                    locations.push(DagSpacetimeLocation {
                        node,
                        qubits: qubits.clone(),
                        before: false,
                        gate_type: gate.gate_type,
                    });
                } else {
                    locations.push(DagSpacetimeLocation {
                        node,
                        qubits: qubits.clone(),
                        before: true,
                        gate_type: gate.gate_type,
                    });
                    locations.push(DagSpacetimeLocation {
                        node,
                        qubits,
                        before: false,
                        gate_type: gate.gate_type,
                    });
                }
            }
        }

        locations
    }

    /// Extract measurements from the propagator.
    fn extract_measurements(&self, propagator: &DagPropagator<'_>) -> Vec<(usize, usize, u8)> {
        let mut measurements = Vec::new();

        for &node in propagator.topo_order() {
            if let Some(gate) = propagator.gate(node) {
                let basis = match gate.gate_type {
                    pecos_quantum::GateType::MZ | pecos_quantum::GateType::MeasureFree => 0,
                    _ => continue,
                };

                for qubit in &gate.qubits {
                    measurements.push((node, qubit.index(), basis));
                }
            }
        }

        measurements
    }

    /// Propagate backward from all detectors.
    fn propagate_detectors(
        &self,
        propagator: &DagPropagator<'_>,
        info: &MeasurementInfo,
        detectors: &[DetectorDef],
        recorder: &mut CompoundRecorder,
    ) {
        let max_node = propagator.max_node();
        let max_qubit = propagator.max_qubit();

        let mut visited = vec![false; max_node + 1];
        let mut active_qubits = vec![false; max_qubit + 1];
        let mut heap = BinaryHeap::new();

        for (det_idx, detector) in detectors.iter().enumerate() {
            // Build combined Pauli from all measurements in the detector
            let mut combined_prop = PauliProp::new();

            for &meas_idx in &detector.measurement_indices {
                // Find the node and qubit for this measurement
                if let Some(node) = info
                    .node_to_meas_idx
                    .iter()
                    .position(|&opt| opt == Some(meas_idx))
                    && let Some(gate) = propagator.gate(node)
                {
                    for qubit in &gate.qubits {
                        // Z-basis measurement means we propagate Z
                        combined_prop.add_z(qubit.index());
                    }
                }
            }

            // Propagate the combined observable backward
            self.propagate_observable(
                propagator,
                &combined_prop,
                det_idx,
                true, // is_detector
                recorder,
                &mut visited,
                &mut active_qubits,
                &mut heap,
            );
        }
    }

    /// Propagate backward from logical operators.
    fn propagate_logicals(&self, propagator: &DagPropagator<'_>, recorder: &mut CompoundRecorder) {
        let max_node = propagator.max_node();
        let max_qubit = propagator.max_qubit();

        let mut visited = vec![false; max_node + 1];
        let mut active_qubits = vec![false; max_qubit + 1];
        let mut heap = BinaryHeap::new();
        let mut logical_idx = 0;

        // Logical X (product of X on specified qubits) - sensitive to Z errors
        if !self.logical_x_qubits.is_empty() {
            let mut prop = PauliProp::new();
            for &q in &self.logical_x_qubits {
                prop.add_x(q);
            }

            self.propagate_observable(
                propagator,
                &prop,
                logical_idx,
                false, // is_detector (this is a logical)
                recorder,
                &mut visited,
                &mut active_qubits,
                &mut heap,
            );
            logical_idx += 1;
        }

        // Logical Z (product of Z on specified qubits) - sensitive to X errors
        if !self.logical_z_qubits.is_empty() {
            let mut prop = PauliProp::new();
            for &q in &self.logical_z_qubits {
                prop.add_z(q);
            }

            self.propagate_observable(
                propagator,
                &prop,
                logical_idx,
                false, // is_detector (this is a logical)
                recorder,
                &mut visited,
                &mut active_qubits,
                &mut heap,
            );
        }
    }

    /// Propagate a single observable backward and record influences.
    #[allow(clippy::too_many_arguments)]
    fn propagate_observable(
        &self,
        propagator: &DagPropagator<'_>,
        initial_prop: &PauliProp,
        target_idx: usize,
        is_detector: bool,
        recorder: &mut CompoundRecorder,
        visited: &mut [bool],
        active_qubits: &mut [bool],
        heap: &mut BinaryHeap<(usize, usize)>,
    ) {
        // Clear work arrays
        visited.fill(false);
        active_qubits.fill(false);
        heap.clear();

        let mut prop = initial_prop.clone();

        // Initialize active qubits from the observable
        for (q, is_active) in active_qubits.iter_mut().enumerate() {
            if prop.contains_x(q) || prop.contains_z(q) {
                *is_active = true;

                // Add all gates on this qubit to the heap
                for (topo_pos, node) in propagator.qubit_gates_backward(q) {
                    if !visited[node] {
                        visited[node] = true;
                        heap.push((topo_pos, node));
                    }
                }
            }
        }

        // Build location index for recording
        let loc_map = self.build_location_map(propagator);

        // Process gates in reverse topological order
        while let Some((_, node)) = heap.pop() {
            if let Some(gate) = propagator.gate(node) {
                // Record influences at before=false location
                if let Some(&loc_idx) = loc_map.get(&(node, false)) {
                    self.record_influence(&prop, loc_idx, target_idx, is_detector, recorder);
                }

                // Track which qubits were active before the gate
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() < active_qubits.len() {
                        was_active[j] = active_qubits[q.index()];
                    }
                }

                // Apply gate backward
                apply_gate(&mut prop, gate, Direction::Backward);

                // Record influences at before=true location
                if let Some(&loc_idx) = loc_map.get(&(node, true)) {
                    self.record_influence(&prop, loc_idx, target_idx, is_detector, recorder);
                }

                // Check if Pauli spread to new qubits
                let node_topo_pos = propagator.topo_position(node);
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx < active_qubits.len() {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];

                        if now_active && !was {
                            // Pauli spread to this qubit - add its gates
                            active_qubits[idx] = true;
                            for (topo_pos, pred_node) in propagator.qubit_gates_backward(idx) {
                                if topo_pos < node_topo_pos && !visited[pred_node] {
                                    visited[pred_node] = true;
                                    heap.push((topo_pos, pred_node));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Build a map from (node, before) to location index.
    fn build_location_map(
        &self,
        propagator: &DagPropagator<'_>,
    ) -> std::collections::HashMap<(usize, bool), usize> {
        let mut map = std::collections::HashMap::new();
        let mut loc_idx = 0;

        for &node in propagator.topo_order() {
            if let Some(gate) = propagator.gate(node) {
                let is_measurement = matches!(
                    gate.gate_type,
                    pecos_quantum::GateType::MZ | pecos_quantum::GateType::MeasureFree
                );
                let is_prep = matches!(
                    gate.gate_type,
                    pecos_quantum::GateType::PZ | pecos_quantum::GateType::QAlloc
                );

                if is_measurement {
                    map.insert((node, true), loc_idx);
                    loc_idx += 1;
                } else if is_prep {
                    map.insert((node, false), loc_idx);
                    loc_idx += 1;
                } else {
                    map.insert((node, true), loc_idx);
                    loc_idx += 1;
                    map.insert((node, false), loc_idx);
                    loc_idx += 1;
                }
            }
        }

        map
    }

    /// Record influence of a fault at a location on a target (detector or logical).
    fn record_influence(
        &self,
        prop: &PauliProp,
        loc_idx: usize,
        target_idx: usize,
        is_detector: bool,
        recorder: &mut CompoundRecorder,
    ) {
        // Check each Pauli type
        for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
            if self.fault_anticommutes(prop, pauli) {
                if is_detector {
                    recorder.record_detector(loc_idx, pauli, target_idx as u32);
                } else {
                    recorder.record_logical(loc_idx, pauli, target_idx as u32);
                }
            }
        }
    }

    /// Check if a fault Pauli anticommutes with the propagated observable.
    fn fault_anticommutes(&self, prop: &PauliProp, fault: Pauli) -> bool {
        let mut anticom_count = 0;

        match fault {
            Pauli::I => return false,
            Pauli::X => {
                // X fault anticommutes with Z component
                anticom_count += prop.get_z_qubits().len();
            }
            Pauli::Z => {
                // Z fault anticommutes with X component
                anticom_count += prop.get_x_qubits().len();
            }
            Pauli::Y => {
                // Y fault anticommutes with both X and Z
                anticom_count += prop.get_x_qubits().len();
                anticom_count += prop.get_z_qubits().len();
            }
        }

        anticom_count % 2 == 1
    }
}

/// Information about measurements from symbolic simulation.
struct MeasurementInfo {
    history: pecos_qsim::symbolic_sparse_stab::MeasurementHistory,
    node_to_meas_idx: Vec<Option<usize>>,
    #[allow(dead_code)]
    num_measurements: usize,
}

/// Definition of a detector as XOR of measurements.
struct DetectorDef {
    /// Measurement indices that XOR together
    measurement_indices: SmallVec<[usize; 4]>,
    /// Expected value (false=0, true=1) in noiseless case
    #[allow(dead_code)]
    expected_value: bool,
}

/// Recorder for compound detector propagation.
struct CompoundRecorder {
    num_locations: usize,
    #[allow(dead_code)]
    num_detectors: usize,
    #[allow(dead_code)]
    num_logicals: usize,

    // Buckets for detector influences [loc_idx][pauli] -> Vec<detector_idx>
    detector_x: Vec<Vec<u32>>,
    detector_y: Vec<Vec<u32>>,
    detector_z: Vec<Vec<u32>>,

    // Buckets for logical influences
    logical_x: Vec<Vec<u32>>,
    logical_y: Vec<Vec<u32>>,
    logical_z: Vec<Vec<u32>>,
}

impl CompoundRecorder {
    fn new(num_locations: usize, num_detectors: usize, num_logicals: usize) -> Self {
        Self {
            num_locations,
            num_detectors,
            num_logicals,
            detector_x: vec![Vec::new(); num_locations],
            detector_y: vec![Vec::new(); num_locations],
            detector_z: vec![Vec::new(); num_locations],
            logical_x: vec![Vec::new(); num_locations],
            logical_y: vec![Vec::new(); num_locations],
            logical_z: vec![Vec::new(); num_locations],
        }
    }

    fn record_detector(&mut self, loc_idx: usize, pauli: Pauli, detector_idx: u32) {
        if loc_idx >= self.num_locations {
            return;
        }
        match pauli {
            Pauli::X => self.detector_x[loc_idx].push(detector_idx),
            Pauli::Y => self.detector_y[loc_idx].push(detector_idx),
            Pauli::Z => self.detector_z[loc_idx].push(detector_idx),
            Pauli::I => {}
        }
    }

    fn record_logical(&mut self, loc_idx: usize, pauli: Pauli, logical_idx: u32) {
        if loc_idx >= self.num_locations {
            return;
        }
        match pauli {
            Pauli::X => self.logical_x[loc_idx].push(logical_idx),
            Pauli::Y => self.logical_y[loc_idx].push(logical_idx),
            Pauli::Z => self.logical_z[loc_idx].push(logical_idx),
            Pauli::I => {}
        }
    }

    fn into_soa(self) -> super::propagator::dag::InfluencesSoA {
        use super::propagator::dag::InfluencesSoA;

        let mut soa = InfluencesSoA::with_capacity(self.num_locations);

        for loc_idx in 0..self.num_locations {
            // Add detector influences
            for &det in &self.detector_x[loc_idx] {
                soa.detectors_x.push(det);
            }
            for &det in &self.detector_y[loc_idx] {
                soa.detectors_y.push(det);
            }
            for &det in &self.detector_z[loc_idx] {
                soa.detectors_z.push(det);
            }

            // Add logical influences
            for &log in &self.logical_x[loc_idx] {
                soa.logicals_x.push(log);
            }
            for &log in &self.logical_y[loc_idx] {
                soa.logicals_y.push(log);
            }
            for &log in &self.logical_z[loc_idx] {
                soa.logicals_z.push(log);
            }

            soa.finish_location();
        }

        soa
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_quantum::DagCircuit;

    #[test]
    fn test_simple_circuit() {
        // Simple circuit: prep, H, measure
        let mut dag = DagCircuit::new();
        dag.pz(0);
        dag.h(0);
        dag.mz(0);

        let builder = InfluenceBuilder::new(&dag);
        let map = builder.build();

        // Should have some locations and at least one detector
        assert!(!map.locations.is_empty());
    }

    #[test]
    fn test_syndrome_extraction() {
        // Simple syndrome extraction: 2 data qubits, 1 ancilla
        let mut dag = DagCircuit::new();

        // Prepare ancilla
        dag.pz(2);

        // CNOT from data to ancilla
        dag.cx(0, 2);
        dag.cx(1, 2);

        // Measure ancilla
        dag.mz(2);

        let builder = InfluenceBuilder::new(&dag);
        let map = builder.build();

        assert!(!map.locations.is_empty());
        assert!(!map.measurements.is_empty());
    }

    #[test]
    fn test_repeated_syndrome() {
        // Two rounds of syndrome extraction
        let mut dag = DagCircuit::new();

        // Round 1
        dag.pz(2);
        dag.cx(0, 2);
        dag.cx(1, 2);
        dag.mz(2);

        // Round 2
        dag.pz(2);
        dag.cx(0, 2);
        dag.cx(1, 2);
        dag.mz(2);

        let builder = InfluenceBuilder::new(&dag);
        let map = builder.build();

        // Should have multiple measurements
        assert!(map.measurements.len() >= 2);

        // The second measurement should be deterministic (depends on first)
        // and thus create a proper detector
        assert!(!map.detectors.is_empty());
    }

    #[test]
    fn test_with_logical() {
        let mut dag = DagCircuit::new();
        dag.pz(2);
        dag.cx(0, 2);
        dag.mz(2);

        let builder = InfluenceBuilder::new(&dag).with_logical_z(vec![0]); // Track Z logical on qubit 0

        let map = builder.build();

        // Should track the logical
        assert!(map.influences.max_logical_index().is_some());
    }
}
