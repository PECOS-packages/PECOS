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
//! ```
//! use pecos_qec::fault_tolerance::InfluenceBuilder;
//! use pecos_qec::fault_tolerance::dem_builder::DemSampler;
//! use pecos_quantum::DagCircuit;
//!
//! // Build a syndrome extraction circuit
//! let mut dag = DagCircuit::new();
//! dag.pz(&[2]);
//! dag.cx(&[(0, 2)]);
//! dag.cx(&[(1, 2)]);
//! dag.mz(&[2]);
//!
//! // Build influence map with automatic detector discovery
//! let builder = InfluenceBuilder::new(&dag);
//! let influence_map = builder.build();
//!
//! // Build a fast DemSampler from the influence map
//! let num_locations = influence_map.locations.len();
//! let sampler = DemSampler::from_influence_map(&influence_map, &vec![0.001; num_locations]);
//! let stats = sampler.sample_statistics(100, 42);
//! ```

use super::propagator::dag::{DagFaultInfluenceMap, DagSpacetimeLocation, DemOutputMetadata};
use super::propagator::types::{DetectorId, MeasurementId};
use super::propagator::{DagFaultAnalyzer, DagPropagator, Direction, Pauli, apply_gate};
use pecos_core::QubitId;
use pecos_simulators::{PauliProp, SymbolicSparseStab};
use smallvec::SmallVec;
use std::collections::BinaryHeap;

/// Builder for fault influence maps with proper detector definitions.
///
/// This integrates forward symbolic simulation with backward propagation
/// to create complete influence maps suitable for noisy sampling.
/// Re-export `PauliString` as the type used for Pauli operator tracking.
///
/// All circuit annotations (detectors, observables, operators) are Pauli
/// strings tracked for flipping via backward propagation. The difference
/// is role and readout:
///
/// | Kind | Pauli | Readout | API |
/// |------|-------|---------|-----|
/// | Detector | Z on measured qubits | measurement XOR = 0 | `dag.detector(&[...])` |
/// | Observable | Z on measured qubits | measurement XOR | `dag.observable(&[...])` |
/// | Operator | user-specified | propagation only | `dag.pauli_operator(&[...])` |
pub use pecos_core::PauliString;

pub struct InfluenceBuilder<'a> {
    dag: &'a pecos_quantum::DagCircuit,
    /// Pauli operators to track for flipping, with optional start position.
    /// `None` means propagate from circuit end; `Some(node)` means propagate
    /// from that DAG node's topological position.
    /// (`metadata`, meta-gate node)
    pauli_operators: Vec<(DemOutputMetadata, Option<usize>)>,
}

impl<'a> InfluenceBuilder<'a> {
    /// Create a new influence builder for the given circuit.
    #[must_use]
    pub fn new(dag: &'a pecos_quantum::DagCircuit) -> Self {
        Self {
            dag,
            pauli_operators: Vec::new(),
        }
    }

    /// Add a tracked X operator (X on all specified qubits).
    #[must_use]
    pub fn with_x(mut self, qubits: &[usize]) -> Self {
        self.pauli_operators.push((
            DemOutputMetadata::tracked_operator(PauliString::xs(qubits)),
            None,
        ));
        self
    }

    /// Add a tracked Z operator (Z on all specified qubits).
    #[must_use]
    pub fn with_z(mut self, qubits: &[usize]) -> Self {
        self.pauli_operators.push((
            DemOutputMetadata::tracked_operator(PauliString::zs(qubits)),
            None,
        ));
        self
    }

    /// Add a tracked Y operator (Y on all specified qubits).
    #[must_use]
    pub fn with_y(mut self, qubits: &[usize]) -> Self {
        self.pauli_operators.push((
            DemOutputMetadata::tracked_operator(PauliString::ys(qubits)),
            None,
        ));
        self
    }

    /// Add a Pauli check: track whether this Pauli string flips due to faults.
    ///
    /// Unlike observables (`dag.observable()`), a Pauli check
    /// uses backward propagation to detect flips WITHOUT requiring a measurement.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Check if Y = X_0 * Z_1 * Z_2 flips
    /// builder.with_pauli_operator(PauliString::from_paulis(vec![(0, 1), (1, 3), (2, 3)]))
    /// ```
    #[must_use]
    pub fn with_pauli_operator(mut self, pauli: PauliString) -> Self {
        self.pauli_operators
            .push((DemOutputMetadata::tracked_operator(pauli), None));
        self
    }

    /// Extract observable and tracked-operator annotations from the circuit.
    ///
    /// Observable annotations define logical observables via measurement records.
    /// For backward propagation, each observable is converted to a Z-type Pauli
    /// on the measured qubits, starting from the latest measurement node.
    ///
    /// Operator annotations have a corresponding `PauliOperatorMeta` node
    /// that marks their time position.
    ///
    /// Detector annotations are NOT handled here -- they are processed
    /// by `DemSamplerBuilder::with_circuit_annotations` which maps them
    /// to auto-detected detectors.
    #[must_use]
    pub fn with_circuit_annotations(mut self, circuit: &pecos_quantum::DagCircuit) -> Self {
        // Find PauliOperatorMeta nodes in topological order.
        // The nth meta-gate corresponds to the nth Operator annotation.
        let meta_nodes: Vec<usize> = circuit
            .topological_order()
            .into_iter()
            .filter(|&node| circuit.gate(node).is_some_and(|g| g.gate_type.is_meta()))
            .collect();

        let mut operator_idx = 0;
        for ann in circuit.annotations() {
            match &ann.kind {
                pecos_quantum::AnnotationKind::Observable { measurement_nodes } => {
                    // Convert measurement-based observable to Z-Pauli on measured qubits.
                    // Backward propagation starts from the latest measurement node.
                    let mut qubits = Vec::new();
                    let mut latest_node = None;
                    for &meas_node in measurement_nodes {
                        if let Some(gate) = circuit.gate(meas_node) {
                            for q in &gate.qubits {
                                qubits.push(q.index());
                            }
                        }
                        latest_node = Some(
                            latest_node.map_or(meas_node, |prev: usize| prev.max(meas_node)),
                        );
                    }
                    let pauli = PauliString::zs(&qubits);
                    self.pauli_operators.push((
                        DemOutputMetadata::observable(pauli)
                            .with_optional_label(ann.label.clone()),
                        latest_node,
                    ));
                }
                pecos_quantum::AnnotationKind::Operator => {
                    let meta_node = meta_nodes.get(operator_idx).copied();
                    operator_idx += 1;
                    self.pauli_operators.push((
                        DemOutputMetadata::tracked_operator(ann.pauli.clone())
                            .with_optional_label(ann.label.clone()),
                        meta_node,
                    ));
                }
                pecos_quantum::AnnotationKind::Detector { .. } => {
                    // Detectors handled separately by DemSamplerBuilder
                }
            }
        }
        self
    }

    /// Build the influence map.
    ///
    /// This performs:
    /// 1. Forward symbolic simulation to get measurement correlations
    /// 2. Detector extraction from deterministic measurements
    /// 3. Backward propagation from detectors and DEM outputs
    #[must_use]
    pub fn build(&self) -> DagFaultInfluenceMap {
        // Step 1: Run forward symbolic simulation
        let measurement_info = self.run_symbolic_simulation();

        // Step 2: Extract detectors from deterministic measurements
        let detectors = Self::extract_detectors(&measurement_info);

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
                        sim.h(&[qubits[0]]);
                    }
                    pecos_quantum::GateType::F => {
                        sim.sx(&[qubits[0]]);
                        sim.sz(&[qubits[0]]);
                    }
                    pecos_quantum::GateType::Fdg => {
                        sim.szdg(&[qubits[0]]);
                        sim.sxdg(&[qubits[0]]);
                    }
                    pecos_quantum::GateType::SX => {
                        sim.sx(&[qubits[0]]);
                    }
                    pecos_quantum::GateType::SXdg => {
                        sim.sxdg(&[qubits[0]]);
                    }
                    pecos_quantum::GateType::SY => {
                        sim.sy(&[qubits[0]]);
                    }
                    pecos_quantum::GateType::SYdg => {
                        sim.sydg(&[qubits[0]]);
                    }
                    pecos_quantum::GateType::SZ => {
                        sim.sz(&[qubits[0]]);
                    }
                    pecos_quantum::GateType::SZdg => {
                        sim.szdg(&[qubits[0]]);
                    }
                    pecos_quantum::GateType::X => {
                        sim.x(&[qubits[0]]);
                    }
                    pecos_quantum::GateType::Y => {
                        sim.y(&[qubits[0]]);
                    }
                    pecos_quantum::GateType::Z => {
                        sim.z(&[qubits[0]]);
                    }
                    pecos_quantum::GateType::CX => {
                        sim.cx(&[(qubits[0], qubits[1])]);
                    }
                    pecos_quantum::GateType::CY => {
                        sim.cy(&[(qubits[0], qubits[1])]);
                    }
                    pecos_quantum::GateType::CZ => {
                        sim.cz(&[(qubits[0], qubits[1])]);
                    }
                    pecos_quantum::GateType::SXX => {
                        sim.sxx(&[(qubits[0], qubits[1])]);
                    }
                    pecos_quantum::GateType::SXXdg => {
                        sim.sxxdg(&[(qubits[0], qubits[1])]);
                    }
                    pecos_quantum::GateType::SYY => {
                        sim.syy(&[(qubits[0], qubits[1])]);
                    }
                    pecos_quantum::GateType::SYYdg => {
                        sim.syydg(&[(qubits[0], qubits[1])]);
                    }
                    pecos_quantum::GateType::SZZ => {
                        sim.szz(&[(qubits[0], qubits[1])]);
                    }
                    pecos_quantum::GateType::SZZdg => {
                        sim.szzdg(&[(qubits[0], qubits[1])]);
                    }
                    pecos_quantum::GateType::SWAP => {
                        sim.swap(&[(qubits[0], qubits[1])]);
                    }
                    pecos_quantum::GateType::MZ | pecos_quantum::GateType::MeasureFree => {
                        sim.mz(&[qubits[0]]);
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
    fn extract_detectors(info: &MeasurementInfo) -> Vec<DetectorDef> {
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
        map.locations = Self::extract_locations(propagator);

        // Build measurement node lookup
        let (measurements, meas_ids) = Self::extract_measurements(propagator);
        map.measurements.clone_from(&measurements);
        map.meas_ids = meas_ids;

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

        // Add tracked Pauli operators in their PECOS tracked-op namespace.
        let num_detectors = detectors.len();
        let num_tracked_ops = self.pauli_operators.len();

        // Build the influence structure using backward propagation
        let mut recorder =
            CompoundRecorder::new(map.locations.len(), num_detectors, num_tracked_ops);

        // Propagate from each detector
        Self::propagate_detectors(propagator, info, detectors, &mut recorder);

        // Propagate from tracked Pauli operators.
        self.propagate_tracked_ops(propagator, &mut recorder);

        // Convert to SoA format
        map.influences = recorder.into_soa();

        // Store DEM-output labels
        map.dem_output_labels = self
            .pauli_operators
            .iter()
            .map(|(metadata, _)| metadata.label.clone())
            .collect();
        map.dem_output_metadata = self
            .pauli_operators
            .iter()
            .map(|(metadata, _)| metadata.clone())
            .collect();

        map
    }

    /// Extract fault locations from the propagator.
    fn extract_locations(propagator: &DagPropagator<'_>) -> Vec<DagSpacetimeLocation> {
        let mut locations = Vec::new();

        for &node in propagator.topo_order() {
            if let Some(gate) = propagator.gate(node) {
                // Meta-gates are not physical -- they don't generate faults
                if gate.gate_type.is_meta() {
                    continue;
                }

                let qubits: Vec<QubitId> = gate.qubits.to_vec();

                let is_measurement = matches!(
                    gate.gate_type,
                    pecos_quantum::GateType::MZ | pecos_quantum::GateType::MeasureFree
                );

                // Standard circuit noise model: one fault location per gate.
                //   Measurement: before. All others: after.
                let before = is_measurement;
                for &q in &qubits {
                    // idle_duration() returns a non-negative integer stored as f64;
                    // truncation and sign loss are not a concern.
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let idle_duration = gate.idle_duration() as u64;
                    locations.push(DagSpacetimeLocation {
                        node,
                        qubits: vec![q],
                        before,
                        gate_type: gate.gate_type,
                        idle_duration,
                    });
                }
            }
        }

        locations
    }

    /// Extract measurements from the propagator.
    fn extract_measurements(
        propagator: &DagPropagator<'_>,
    ) -> (Vec<(usize, usize, u8)>, Vec<pecos_core::MeasId>) {
        let mut entries: Vec<(usize, usize, usize, u8, Option<pecos_core::MeasId>)> = Vec::new();

        for &node in propagator.topo_order() {
            if let Some(gate) = propagator.gate(node) {
                let basis = match gate.gate_type {
                    pecos_quantum::GateType::MZ | pecos_quantum::GateType::MeasureFree => 0,
                    _ => continue,
                };

                if !gate.meas_ids.is_empty() {
                    for (i, qubit) in gate.qubits.iter().enumerate() {
                        let mr = gate.meas_ids.get(i).copied();
                        let sort_key = mr.map(|m| m.index()).unwrap_or(usize::MAX);
                        entries.push((sort_key, node, qubit.index(), basis, mr));
                    }
                } else {
                    let topo_pos = propagator.topo_position(node);
                    for qubit in &gate.qubits {
                        entries.push((topo_pos, node, qubit.index(), basis, None));
                    }
                }
            }
        }

        entries.sort_by_key(|&(sort_key, _, qubit, _, _)| (sort_key, qubit));

        let has_meas_ids = entries.iter().any(|(_, _, _, _, mr)| mr.is_some());
        let meas_ids = if has_meas_ids {
            entries
                .iter()
                .map(|(_, _, _, _, mr)| mr.unwrap_or(pecos_core::MeasId(usize::MAX)))
                .collect()
        } else {
            Vec::new()
        };

        let measurements = entries
            .into_iter()
            .map(|(_, node, qubit, basis, _)| (node, qubit, basis))
            .collect();

        (measurements, meas_ids)
    }

    /// Propagate backward from all detectors.
    fn propagate_detectors(
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
                        combined_prop.track_z(&[qubit.index()]);
                    }
                }
            }

            // Propagate the combined observable backward
            Self::propagate_observable(
                propagator,
                &combined_prop,
                det_idx,
                true, // is_detector
                recorder,
                &mut visited,
                &mut active_qubits,
                &mut heap,
                None, // detectors: walk from circuit end
            );
        }
    }

    /// Propagate backward from tracked Pauli operators.
    ///
    /// If a Pauli operator has a corresponding `PauliOperatorMeta` node in the
    /// DAG, propagation starts from that node's topological position. Otherwise
    /// (e.g. operators added via `with_z`/`with_x` without a circuit annotation),
    /// propagation walks from the circuit end.
    fn propagate_tracked_ops(
        &self,
        propagator: &DagPropagator<'_>,
        recorder: &mut CompoundRecorder,
    ) {
        let max_node = propagator.max_node();
        let max_qubit = propagator.max_qubit();

        let mut visited = vec![false; max_node + 1];
        let mut active_qubits = vec![false; max_qubit + 1];
        let mut heap = BinaryHeap::new();

        for (dem_output_idx, (metadata, meta_node)) in self.pauli_operators.iter().enumerate() {
            let mut prop = PauliProp::new();

            for &(pauli, qubit) in metadata.pauli.paulis() {
                use pecos_core::Pauli;
                let q = qubit.index();
                match pauli {
                    Pauli::X => prop.track_x(&[q]),
                    Pauli::Y => prop.track_y(&[q]),
                    Pauli::Z => prop.track_z(&[q]),
                    Pauli::I => {}
                }
            }

            // Resolve the meta-gate node to its topological position.
            // None means no positional bound (walk from circuit end).
            let start_pos = meta_node.map(|node| propagator.topo_position(node));

            Self::propagate_observable(
                propagator,
                &prop,
                dem_output_idx,
                false, // is_detector = false (this is a DEM output)
                recorder,
                &mut visited,
                &mut active_qubits,
                &mut heap,
                start_pos,
            );
        }
    }

    /// Propagate a single observable backward and record influences.
    ///
    /// When `start_topo_pos` is `Some(pos)`, only gates at or before that
    /// topological position are considered. This makes Pauli operator
    /// annotations positional: only faults before the meta-gate affect it.
    #[allow(clippy::too_many_arguments)]
    fn propagate_observable(
        propagator: &DagPropagator<'_>,
        initial_prop: &PauliProp,
        target_idx: usize,
        is_detector: bool,
        recorder: &mut CompoundRecorder,
        visited: &mut [bool],
        active_qubits: &mut [bool],
        heap: &mut BinaryHeap<(usize, usize)>,
        start_topo_pos: Option<usize>,
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

                // Add gates on this qubit to the heap, bounded by start position
                for (topo_pos, node) in propagator.qubit_gates_backward(q) {
                    if start_topo_pos.is_some_and(|max| topo_pos > max) {
                        continue;
                    }
                    if !visited[node] {
                        visited[node] = true;
                        heap.push((topo_pos, node));
                    }
                }
            }
        }

        // Build location index for recording
        let loc_map = Self::build_location_map(propagator);

        // Process gates in reverse topological order
        while let Some((_, node)) = heap.pop() {
            if let Some(gate) = propagator.gate(node) {
                // Record per-qubit influences at before=false location
                if let Some(qubit_locs) = loc_map.get(&(node, false)) {
                    Self::record_influence(&prop, qubit_locs, target_idx, is_detector, recorder);
                }

                // Track which qubits were active before the gate
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() < active_qubits.len() {
                        was_active[j] = active_qubits[q.index()];
                    }
                }

                // Prep gates (PZ/QAlloc) reset the qubit -- kill the Pauli
                // and mark the qubit inactive. Faults before the prep
                // cannot propagate past it.
                let is_prep = matches!(
                    gate.gate_type,
                    pecos_quantum::GateType::PZ | pecos_quantum::GateType::QAlloc
                );
                if is_prep {
                    for q in &gate.qubits {
                        let qi = q.index();
                        // Toggle off X and Z components (XOR to zero)
                        if prop.contains_x(qi) {
                            prop.track_x(&[qi]);
                        }
                        if prop.contains_z(qi) {
                            prop.track_z(&[qi]);
                        }
                        if qi < active_qubits.len() {
                            active_qubits[qi] = false;
                        }
                    }
                    continue; // don't propagate further on these qubits
                }

                // Apply gate backward
                apply_gate(&mut prop, gate, Direction::Backward);

                // Record per-qubit influences at before=true location
                if let Some(qubit_locs) = loc_map.get(&(node, true)) {
                    Self::record_influence(&prop, qubit_locs, target_idx, is_detector, recorder);
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

    /// Build a map from (node, before) to per-qubit location indices.
    fn build_location_map(
        propagator: &DagPropagator<'_>,
    ) -> std::collections::HashMap<(usize, bool), Vec<(usize, usize)>> {
        // (node, before) -> [(qubit_index, loc_idx), ...]
        let mut map: std::collections::HashMap<(usize, bool), Vec<(usize, usize)>> =
            std::collections::HashMap::new();
        let mut loc_idx = 0;

        for &node in propagator.topo_order() {
            if let Some(gate) = propagator.gate(node) {
                if gate.gate_type.is_meta() {
                    continue;
                }

                let is_measurement = matches!(
                    gate.gate_type,
                    pecos_quantum::GateType::MZ | pecos_quantum::GateType::MeasureFree
                );

                let before = is_measurement;
                for q in &gate.qubits {
                    let qi = q.index();
                    map.entry((node, before)).or_default().push((qi, loc_idx));
                    loc_idx += 1;
                }
            }
        }

        map
    }

    /// Record per-qubit influence of a fault at a gate location.
    fn record_influence(
        prop: &PauliProp,
        qubit_locs: &[(usize, usize)], // [(qubit, loc_idx), ...]
        target_idx: usize,
        is_detector: bool,
        recorder: &mut CompoundRecorder,
    ) {
        for &(qubit, loc_idx) in qubit_locs {
            for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
                if Self::fault_anticommutes_qubit(prop, qubit, pauli) {
                    if is_detector {
                        #[allow(clippy::cast_possible_truncation)]
                        recorder.record_detector(loc_idx, pauli, target_idx as u32);
                    } else {
                        #[allow(clippy::cast_possible_truncation)]
                        recorder.record_dem_output(loc_idx, pauli, target_idx as u32);
                    }
                }
            }
        }
    }

    /// Check if a single-qubit fault Pauli anticommutes with the propagated
    /// observable on a specific qubit.
    fn fault_anticommutes_qubit(prop: &PauliProp, qubit: usize, fault: Pauli) -> bool {
        let has_x = prop.contains_x(qubit);
        let has_z = prop.contains_z(qubit);

        match fault {
            Pauli::I => false,
            Pauli::X => has_z,         // X anticommutes with Z
            Pauli::Z => has_x,         // Z anticommutes with X
            Pauli::Y => has_x ^ has_z, // Y anticommutes with X or Z but not both
        }
    }
}

/// Information about measurements from symbolic simulation.
struct MeasurementInfo {
    history: pecos_simulators::symbolic_sparse_stab::MeasurementHistory,
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
    num_tracked_ops: usize,

    // Buckets for detector influences [loc_idx][pauli] -> Vec<detector_idx>
    detector_x: Vec<Vec<u32>>,
    detector_y: Vec<Vec<u32>>,
    detector_z: Vec<Vec<u32>>,

    // Buckets for DEM-output influences.
    dem_output_x: Vec<Vec<u32>>,
    dem_output_y: Vec<Vec<u32>>,
    dem_output_z: Vec<Vec<u32>>,
}

impl CompoundRecorder {
    fn new(num_locations: usize, num_detectors: usize, num_tracked_ops: usize) -> Self {
        Self {
            num_locations,
            num_detectors,
            num_tracked_ops,
            detector_x: vec![Vec::new(); num_locations],
            detector_y: vec![Vec::new(); num_locations],
            detector_z: vec![Vec::new(); num_locations],
            dem_output_x: vec![Vec::new(); num_locations],
            dem_output_y: vec![Vec::new(); num_locations],
            dem_output_z: vec![Vec::new(); num_locations],
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

    fn record_dem_output(&mut self, loc_idx: usize, pauli: Pauli, dem_output_idx: u32) {
        if loc_idx >= self.num_locations {
            return;
        }
        match pauli {
            Pauli::X => self.dem_output_x[loc_idx].push(dem_output_idx),
            Pauli::Y => self.dem_output_y[loc_idx].push(dem_output_idx),
            Pauli::Z => self.dem_output_z[loc_idx].push(dem_output_idx),
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

            // Add DEM-output influences
            for &dem_output in &self.dem_output_x[loc_idx] {
                soa.dem_outputs_x.push(dem_output);
            }
            for &dem_output in &self.dem_output_y[loc_idx] {
                soa.dem_outputs_y.push(dem_output);
            }
            for &dem_output in &self.dem_output_z[loc_idx] {
                soa.dem_outputs_z.push(dem_output);
            }

            soa.finish_location();
        }

        soa
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fault_tolerance::propagator::DemOutputKind;
    use pecos_quantum::DagCircuit;

    #[test]
    fn test_simple_circuit() {
        // Simple circuit: prep, H, measure
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.h(&[0]);
        dag.mz(&[0]);

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
        dag.pz(&[2]);

        // CNOT from data to ancilla
        dag.cx(&[(0, 2)]);
        dag.cx(&[(1, 2)]);

        // Measure ancilla
        dag.mz(&[2]);

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
        dag.pz(&[2]);
        dag.cx(&[(0, 2)]);
        dag.cx(&[(1, 2)]);
        dag.mz(&[2]);

        // Round 2
        dag.pz(&[2]);
        dag.cx(&[(0, 2)]);
        dag.cx(&[(1, 2)]);
        dag.mz(&[2]);

        let builder = InfluenceBuilder::new(&dag);
        let map = builder.build();

        // Should have multiple measurements
        assert!(map.measurements.len() >= 2);

        // The second measurement should be deterministic (depends on first)
        // and thus create a proper detector
        assert!(!map.detectors.is_empty());
    }

    #[test]
    fn test_with_pauli_operator() {
        let mut dag = DagCircuit::new();
        dag.pz(&[2]);
        dag.cx(&[(0, 2)]);
        dag.mz(&[2]);

        let builder = InfluenceBuilder::new(&dag).with_z(&[0]); // Track Z logical on qubit 0

        let map = builder.build();

        // Should track the logical
        assert!(map.influences.max_dem_output_index().is_some());
    }

    #[test]
    fn test_dem_output_metadata_accepts_pauli_string_and_normalizes_phase() {
        use pecos_core::{Pauli, QuarterPhase};

        let pauli =
            PauliString::from_paulis_with_phase(QuarterPhase::MinusI, &[Pauli::X, Pauli::Z]);
        let metadata = DemOutputMetadata::tracked_operator(pauli).with_label("xz");

        assert_eq!(metadata.kind, DemOutputKind::TrackedOperator);
        assert_eq!(metadata.label.as_deref(), Some("xz"));
        assert_eq!(metadata.pauli.phase(), QuarterPhase::PlusOne);
        assert_eq!(metadata.pauli.to_sparse_str(), "+X0 Z1");
    }

    #[test]
    fn test_circuit_annotation_dem_output_metadata_tracks_observables_and_operators() {
        use pecos_core::pauli::constructors::X;

        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.h(&[0]);
        let meas = dag.mz(&[0]);
        dag.observable_labeled("record_obs", &[meas[0]]);
        dag.pauli_operator_labeled("track_x", X(0));

        let map = InfluenceBuilder::new(&dag)
            .with_circuit_annotations(&dag)
            .build();

        // 1 observable (record_obs) + 1 tracked operator (track_x) = 2 DEM outputs
        assert_eq!(map.num_dem_outputs(), 1, "1 observable");
        assert_eq!(map.num_tracked_ops(), 1, "1 tracked operator");
        assert_eq!(map.dem_output_metadata.len(), 2);

        // Observable comes first (annotations are processed in order)
        assert_eq!(
            map.dem_output_metadata[0].kind,
            DemOutputKind::Observable
        );
        assert_eq!(map.dem_output_metadata[0].label.as_deref(), Some("record_obs"));

        // Tracked operator second
        assert_eq!(
            map.dem_output_metadata[1].kind,
            DemOutputKind::TrackedOperator
        );
        assert_eq!(map.dem_output_metadata[1].label.as_deref(), Some("track_x"));
        assert_eq!(map.dem_output_metadata[1].pauli.to_sparse_str(), "+X0");
    }
}
