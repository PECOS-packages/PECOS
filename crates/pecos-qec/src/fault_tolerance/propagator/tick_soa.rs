//! Optimized DOD-based tick fault analyzer using `TickCircuitSoA`.
//!
//! This module provides [`TickFaultAnalyzerSoA`] which leverages the Structure of Arrays
//! layout of [`TickCircuitSoA`] for more cache-efficient fault analysis.
//!
//! # Optimizations
//!
//! 1. **Raw index access**: Uses u32 indices instead of `GateId` validation
//! 2. **Bitset for visited tracking**: O(1) membership check instead of `Vec::contains`
//! 3. **Pre-computed tick indexes**: O(1) lookup for gates in each tick
//! 4. **Sorted qubit gates**: Gates per qubit sorted by tick for efficient backward traversal
//! 5. **Direct array access**: Skips Option-returning methods in hot loops

use super::SpacetimeLocation;
use super::types::{DetectorId, FaultInfluence, FaultInfluenceMap, LogicalId, MeasurementId};
use pecos_core::gate_type::GateType;
use pecos_quantum::tick_circuit_soa::TickCircuitSoA;
use pecos_simulators::PauliProp;

// ============================================================================
// Work Buffers for Reuse
// ============================================================================

/// Reusable work buffers for fault analysis.
#[derive(Debug, Clone)]
pub struct AnalyzerWorkBuffers {
    /// Bitset for tracking active qubits
    active_qubits: Vec<bool>,
    /// Bitset for tracking processed gates in current tick
    processed_gates: Vec<bool>,
    /// Temporary storage for gates to process
    gates_to_process: Vec<u32>,
}

impl AnalyzerWorkBuffers {
    /// Creates work buffers sized for the circuit.
    pub fn new(max_qubit: usize, max_gate: usize) -> Self {
        Self {
            active_qubits: vec![false; max_qubit + 1],
            processed_gates: vec![false; max_gate + 1],
            gates_to_process: Vec::with_capacity(64),
        }
    }

    /// Clears buffers for reuse.
    pub fn clear(&mut self) {
        self.active_qubits.fill(false);
        self.processed_gates.fill(false);
        self.gates_to_process.clear();
    }

    /// Resizes buffers if needed.
    #[allow(dead_code)]
    pub fn ensure_capacity(&mut self, max_qubit: usize, max_gate: usize) {
        if self.active_qubits.len() <= max_qubit {
            self.active_qubits.resize(max_qubit + 1, false);
        }
        if self.processed_gates.len() <= max_gate {
            self.processed_gates.resize(max_gate + 1, false);
        }
    }
}

// ============================================================================
// Optimized SOA-Based Fault Analyzer
// ============================================================================

/// Optimized fault analyzer for `TickCircuitSoA`.
///
/// Uses raw indices and bitsets for minimal overhead in hot paths.
pub struct TickFaultAnalyzerSoA<'a> {
    circuit: &'a TickCircuitSoA,
    /// Fault locations extracted from the circuit.
    locations: Vec<SpacetimeLocation>,
    /// Pre-computed index: tick -> (`location_index`, before) pairs
    tick_locations: Vec<Vec<(usize, bool)>>,
}

impl<'a> TickFaultAnalyzerSoA<'a> {
    /// Creates a new analyzer for the given `SoA` circuit.
    #[must_use]
    pub fn new(circuit: &'a TickCircuitSoA) -> Self {
        let locations = Self::extract_spacetime_locations(circuit);

        // Build tick index for O(1) lookup
        let num_ticks = circuit.num_ticks();
        let mut tick_locations = vec![Vec::new(); num_ticks + 1];
        for (idx, loc) in locations.iter().enumerate() {
            if loc.tick < tick_locations.len() {
                tick_locations[loc.tick].push((idx, loc.before));
            }
        }

        Self {
            circuit,
            locations,
            tick_locations,
        }
    }

    /// Extracts spacetime locations using raw index access.
    fn extract_spacetime_locations(circuit: &TickCircuitSoA) -> Vec<SpacetimeLocation> {
        let mut locations = Vec::new();
        let storage = &circuit.storage;

        for idx in 0..storage.slot_count() {
            if !storage.is_occupied(idx) {
                continue;
            }

            let gate_type = storage.type_unchecked(idx);
            let qubits = storage.qubits_unchecked(idx).to_vec();
            let tick = storage.tick_id_unchecked(idx) as usize;

            // Before location (fault before gate)
            locations.push(SpacetimeLocation {
                tick,
                qubits: qubits.clone(),
                before: true,
                gate_type,
                gate_index: idx,
            });

            // After location for most gates (except prep which resets)
            if !matches!(gate_type, GateType::PZ | GateType::QAlloc) {
                locations.push(SpacetimeLocation {
                    tick,
                    qubits,
                    before: false,
                    gate_type,
                    gate_index: idx,
                });
            }
        }

        locations
    }

    /// Returns the fault locations.
    #[must_use]
    pub fn locations(&self) -> &[SpacetimeLocation] {
        &self.locations
    }

    /// Builds the complete fault influence map.
    #[must_use]
    pub fn build_influence_map(&self) -> FaultInfluenceMap {
        self.build_influence_map_with_logicals(&[])
    }

    /// Builds the fault influence map with logical operator tracking.
    #[must_use]
    pub fn build_influence_map_with_logicals(
        &self,
        logicals: &[(&[usize], &[usize])],
    ) -> FaultInfluenceMap {
        let mut map = FaultInfluenceMap::new();

        // Extract all measurements from the circuit
        let measurements = self.extract_measurements();
        map.measurements.clone_from(&measurements);

        // Create simple detectors (one per measurement)
        for m in &measurements {
            map.detectors.push(DetectorId::single(*m));
        }

        // Create logical IDs
        for (i, _) in logicals.iter().enumerate() {
            map.logicals.push(LogicalId {
                logical_qubit: i,
                observable: 0,
            });
        }

        // Initialize influence for each fault location
        for loc in &self.locations {
            map.influences
                .insert(loc.clone(), FaultInfluence::default());
        }

        // Create work buffers
        let max_qubit = self.circuit.max_qubit();
        let max_gate = self.circuit.storage.slot_count();
        let mut buffers = AnalyzerWorkBuffers::new(max_qubit, max_gate);

        // Backward propagate from each measurement
        for measurement in &measurements {
            self.propagate_from_measurement_optimized(measurement, &mut map, &mut buffers);
        }

        // Backward propagate from each logical operator
        for (i, (x_pos, z_pos)) in logicals.iter().enumerate() {
            let logical_id = LogicalId {
                logical_qubit: i,
                observable: 0,
            };
            self.propagate_from_logical_optimized(
                x_pos,
                z_pos,
                &logical_id,
                &mut map,
                &mut buffers,
            );
        }

        // Build reverse maps
        Self::build_reverse_maps(&mut map);

        map
    }

    /// Extracts all measurements using raw index access.
    fn extract_measurements(&self) -> Vec<MeasurementId> {
        let mut measurements = Vec::new();
        let storage = &self.circuit.storage;

        for idx in 0..storage.slot_count() {
            if !storage.is_occupied(idx) {
                continue;
            }

            let gate_type = storage.type_unchecked(idx);
            let basis = match gate_type {
                GateType::MZ | GateType::MeasureFree => 0, // Z-basis
                _ => continue,
            };

            let tick = storage.tick_id_unchecked(idx) as usize;
            let qubits = storage.qubits_unchecked(idx);

            for qubit in qubits {
                measurements.push(MeasurementId {
                    tick,
                    qubit: qubit.index(),
                    basis,
                });
            }
        }

        measurements
    }

    /// Optimized backward propagation from a measurement.
    fn propagate_from_measurement_optimized(
        &self,
        measurement: &MeasurementId,
        map: &mut FaultInfluenceMap,
        buffers: &mut AnalyzerWorkBuffers,
    ) {
        buffers.clear();

        let initial_pauli = if measurement.basis == 0 { 3u8 } else { 1u8 };

        let mut prop = PauliProp::new();
        match initial_pauli {
            1 => prop.track_x(&[measurement.qubit]),
            3 => prop.track_z(&[measurement.qubit]),
            _ => {}
        }

        let detector = DetectorId::single(*measurement);

        // Mark initial qubit as active
        if measurement.qubit < buffers.active_qubits.len() {
            buffers.active_qubits[measurement.qubit] = true;
        }

        // Check faults at the measurement tick (before=true only)
        self.record_influences_at_tick_filtered(
            measurement.tick,
            &prop,
            &detector,
            None,
            map,
            true,
        );

        // Propagate backward through ticks
        for tick_idx in (0..measurement.tick).rev() {
            // Check before=false locations
            self.record_influences_at_tick_filtered(tick_idx, &prop, &detector, None, map, false);

            // Apply gates at this tick backward using optimized sparse traversal
            self.apply_tick_backward_optimized(tick_idx, &mut prop, buffers);

            // Check before=true locations
            self.record_influences_at_tick_filtered(tick_idx, &prop, &detector, None, map, true);
        }
    }

    /// Optimized backward gate application using bitsets and pre-computed indexes.
    #[inline]
    fn apply_tick_backward_optimized(
        &self,
        tick_idx: usize,
        prop: &mut PauliProp,
        buffers: &mut AnalyzerWorkBuffers,
    ) {
        let storage = &self.circuit.storage;

        // Clear processed gates bitset for this tick
        buffers.gates_to_process.clear();

        // Get gates in this tick directly from pre-computed index
        let tick_gates = self.circuit.gates_in_tick_raw(tick_idx);

        // Find gates that touch active qubits
        for &gate_idx in tick_gates {
            let idx = gate_idx as usize;
            if !storage.is_occupied(idx) {
                continue;
            }

            let qubits = storage.qubits_unchecked(idx);

            // Check if any qubit is active
            let touches_active = qubits.iter().any(|q| {
                let qi = q.index();
                qi < buffers.active_qubits.len() && buffers.active_qubits[qi]
            });

            if touches_active {
                buffers.gates_to_process.push(gate_idx);
            }
        }

        // Apply each gate backward
        // Note: We iterate by index to avoid borrow conflict
        let num_gates = buffers.gates_to_process.len();
        for i in 0..num_gates {
            let gate_idx = buffers.gates_to_process[i] as usize;
            self.apply_gate_backward_raw(gate_idx, prop, buffers);
        }
    }

    /// Apply a single gate backward using raw index access.
    #[inline]
    fn apply_gate_backward_raw(
        &self,
        idx: usize,
        prop: &mut PauliProp,
        buffers: &mut AnalyzerWorkBuffers,
    ) {
        let storage = &self.circuit.storage;
        let gate_type = storage.type_unchecked(idx);
        let qubits = storage.qubits_unchecked(idx);

        match gate_type {
            GateType::CX if qubits.len() >= 2 => {
                let control = qubits[0].index();
                let target = qubits[1].index();

                let ctrl_x = prop.contains_x(control);
                let tgt_z = prop.contains_z(target);

                if ctrl_x {
                    prop.track_x(&[target]);
                }
                if tgt_z {
                    prop.track_z(&[control]);
                }

                // Update active qubits
                Self::update_active_qubit(control, prop, buffers);
                Self::update_active_qubit(target, prop, buffers);
            }

            GateType::CZ if qubits.len() >= 2 => {
                let q0 = qubits[0].index();
                let q1 = qubits[1].index();

                let x0 = prop.contains_x(q0);
                let x1 = prop.contains_x(q1);

                if x0 {
                    prop.track_z(&[q1]);
                }
                if x1 {
                    prop.track_z(&[q0]);
                }

                Self::update_active_qubit(q0, prop, buffers);
                Self::update_active_qubit(q1, prop, buffers);
            }

            GateType::H => {
                if let Some(qid) = qubits.first() {
                    let q = qid.index();
                    let has_x = prop.contains_x(q);
                    let has_z = prop.contains_z(q);

                    if has_x && !has_z {
                        prop.track_x(&[q]);
                        prop.track_z(&[q]);
                    } else if has_z && !has_x {
                        prop.track_z(&[q]);
                        prop.track_x(&[q]);
                    }

                    Self::update_active_qubit(q, prop, buffers);
                }
            }

            GateType::SZ | GateType::SZdg => {
                if let Some(qid) = qubits.first() {
                    let q = qid.index();
                    let has_x = prop.contains_x(q);

                    if has_x {
                        prop.track_z(&[q]);
                    }

                    Self::update_active_qubit(q, prop, buffers);
                }
            }

            GateType::PZ | GateType::QAlloc => {
                // Preparation resets - kill the Pauli
                for qid in qubits {
                    let q = qid.index();
                    if prop.contains_x(q) {
                        prop.track_x(&[q]);
                    }
                    if prop.contains_z(q) {
                        prop.track_z(&[q]);
                    }
                    if q < buffers.active_qubits.len() {
                        buffers.active_qubits[q] = false;
                    }
                }
            }

            // Pauli gates (X,Y,Z), Measure, MeasureFree, and other gates - no change
            _ => {}
        }
    }

    /// Update active qubit status.
    #[inline]
    fn update_active_qubit(q: usize, prop: &PauliProp, buffers: &mut AnalyzerWorkBuffers) {
        if q < buffers.active_qubits.len() {
            buffers.active_qubits[q] = prop.contains_x(q) || prop.contains_z(q);
        }
    }

    /// Optimized backward propagation from a logical operator.
    fn propagate_from_logical_optimized(
        &self,
        x_positions: &[usize],
        z_positions: &[usize],
        logical_id: &LogicalId,
        map: &mut FaultInfluenceMap,
        buffers: &mut AnalyzerWorkBuffers,
    ) {
        buffers.clear();

        let mut prop = PauliProp::new();

        for &q in x_positions {
            prop.track_x(&[q]);
            if q < buffers.active_qubits.len() {
                buffers.active_qubits[q] = true;
            }
        }
        for &q in z_positions {
            prop.track_z(&[q]);
            if q < buffers.active_qubits.len() {
                buffers.active_qubits[q] = true;
            }
        }

        let dummy_detector = DetectorId::single(MeasurementId {
            tick: 0,
            qubit: 0,
            basis: 0,
        });

        let num_ticks = self.circuit.num_ticks();
        for tick_idx in (0..num_ticks).rev() {
            self.record_influences_at_tick_filtered(
                tick_idx,
                &prop,
                &dummy_detector,
                Some(logical_id),
                map,
                false,
            );

            self.apply_tick_backward_optimized(tick_idx, &mut prop, buffers);

            self.record_influences_at_tick_filtered(
                tick_idx,
                &prop,
                &dummy_detector,
                Some(logical_id),
                map,
                true,
            );
        }
    }

    /// Records influences at a tick.
    #[inline]
    fn record_influences_at_tick_filtered(
        &self,
        tick_idx: usize,
        prop: &PauliProp,
        detector: &DetectorId,
        logical: Option<&LogicalId>,
        map: &mut FaultInfluenceMap,
        only_before: bool,
    ) {
        if tick_idx >= self.tick_locations.len() {
            return;
        }

        for &(loc_idx, before) in &self.tick_locations[tick_idx] {
            if before != only_before {
                continue;
            }

            let loc = &self.locations[loc_idx];

            for (qubit_idx, qubit) in loc.qubits.iter().enumerate() {
                let q = qubit.index();

                let obs_x = prop.contains_x(q);
                let obs_z = prop.contains_z(q);

                if let Some(influence) = map.influences.get_mut(loc) {
                    // X fault anticommutes with Z or Y observable
                    if obs_z {
                        if let Some(log) = logical {
                            influence.logical_flips[1].push(*log);
                        } else {
                            influence.detector_flips[1].push(detector.clone());
                            influence.measurement_flips[1]
                                .extend(detector.measurements.iter().copied());
                            influence
                                .per_qubit_detector_flips
                                .entry((qubit_idx, 1))
                                .or_default()
                                .push(detector.clone());
                        }
                    }

                    // Z fault anticommutes with X or Y observable
                    if obs_x {
                        if let Some(log) = logical {
                            influence.logical_flips[3].push(*log);
                        } else {
                            influence.detector_flips[3].push(detector.clone());
                            influence.measurement_flips[3]
                                .extend(detector.measurements.iter().copied());
                            influence
                                .per_qubit_detector_flips
                                .entry((qubit_idx, 3))
                                .or_default()
                                .push(detector.clone());
                        }
                    }

                    // Y fault anticommutes with any non-identity observable
                    if obs_x || obs_z {
                        if let Some(log) = logical {
                            influence.logical_flips[2].push(*log);
                        } else {
                            influence.detector_flips[2].push(detector.clone());
                            influence.measurement_flips[2]
                                .extend(detector.measurements.iter().copied());
                            influence
                                .per_qubit_detector_flips
                                .entry((qubit_idx, 2))
                                .or_default()
                                .push(detector.clone());
                        }
                    }
                }
            }
        }
    }

    /// Builds reverse maps.
    fn build_reverse_maps(map: &mut FaultInfluenceMap) {
        for (loc, influence) in &map.influences {
            for (pauli, detectors) in influence.detector_flips.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)] // Pauli index 0..2
                let pauli_u8 = pauli as u8;
                for detector in detectors {
                    map.detector_to_faults
                        .entry(detector.clone())
                        .or_default()
                        .push((loc.clone(), pauli_u8));
                }
            }

            for (pauli, logicals) in influence.logical_flips.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)] // Pauli index 0..2
                let pauli_u8 = pauli as u8;
                for logical in logicals {
                    map.logical_to_faults
                        .entry(*logical)
                        .or_default()
                        .push((loc.clone(), pauli_u8));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_quantum::tick_circuit_soa::TickCircuitSoABuilder;

    #[test]
    fn test_basic_analysis() {
        let mut builder = TickCircuitSoABuilder::new();
        builder
            .tick()
            .pz(&[0, 1])
            .tick()
            .h(&[0])
            .tick()
            .cx(&[(0, 1)])
            .tick()
            .mz(&[0, 1]);

        let circuit = builder.build();
        let analyzer = TickFaultAnalyzerSoA::new(&circuit);

        assert!(!analyzer.locations().is_empty());

        let map = analyzer.build_influence_map();

        assert!(!map.measurements.is_empty());
        assert!(!map.detectors.is_empty());
    }

    #[test]
    fn test_sparse_traversal() {
        let mut builder = TickCircuitSoABuilder::new();
        builder
            .tick()
            .pz(&[0, 1, 2, 3])
            .tick()
            .h(&[0])
            .h(&[2])
            .tick()
            .cx(&[(0, 1)])
            .cx(&[(2, 3)])
            .tick()
            .mz(&[0, 1, 2, 3]);

        let circuit = builder.build();
        let analyzer = TickFaultAnalyzerSoA::new(&circuit);

        let map = analyzer.build_influence_map();

        assert_eq!(map.measurements.len(), 4);
        assert_eq!(map.detectors.len(), 4);
    }

    #[test]
    fn test_logical_propagation() {
        let mut builder = TickCircuitSoABuilder::new();
        builder
            .tick()
            .pz(&[0, 1])
            .tick()
            .h(&[0])
            .tick()
            .cx(&[(0, 1)])
            .tick()
            .mz(&[0, 1]);

        let circuit = builder.build();
        let analyzer = TickFaultAnalyzerSoA::new(&circuit);

        let logicals = [(&[] as &[usize], &[1usize] as &[usize])];
        let map = analyzer.build_influence_map_with_logicals(&logicals);

        assert_eq!(map.logicals.len(), 1);
    }
}
