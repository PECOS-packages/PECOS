//! Optimized tick fault analyzer using batched `TickCircuit` access.
//!
//! This module provides [`TickFaultAnalyzerBatched`] which uses the batched
//! full-fidelity command view of [`TickCircuit`] for cache-efficient fault
//! analysis without requiring a converted circuit representation.
//!
//! # Optimizations
//!
//! 1. **Raw index access**: Uses local flattened gate indices
//! 2. **Bitset for visited tracking**: O(1) membership check instead of `Vec::contains`
//! 3. **Pre-computed tick indexes**: O(1) lookup for gates in each tick
//! 4. **Direct array access**: Skips Option-returning methods in hot loops

use super::SpacetimeLocation;
use super::types::{DetectorId, FaultInfluence, FaultInfluenceMap, MeasurementId, TrackedPauliId};
use pecos_core::{QubitId, gate_type::GateType};
use pecos_quantum::TickCircuit;
use pecos_simulators::{CliffordGateable, PauliProp};

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
    gates_to_process: Vec<usize>,
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
// Optimized Batched Tick Fault Analyzer
// ============================================================================

#[derive(Debug, Clone)]
struct AnalyzerGate {
    tick: usize,
    gate_type: GateType,
    qubits: Vec<QubitId>,
}

/// Optimized fault analyzer for `TickCircuit`.
///
/// Uses raw indices and bitsets for minimal overhead in hot paths.
pub struct TickFaultAnalyzerBatched<'a> {
    circuit: &'a TickCircuit,
    /// Flattened full-fidelity gate command view.
    gates: Vec<AnalyzerGate>,
    /// Pre-computed index: tick -> gate indices
    tick_gates: Vec<Vec<usize>>,
    /// Maximum qubit index seen.
    max_qubit: usize,
    /// Fault locations extracted from the circuit.
    locations: Vec<SpacetimeLocation>,
    /// Pre-computed index: tick -> (`location_index`, before) pairs
    tick_locations: Vec<Vec<(usize, bool)>>,
}

impl<'a> TickFaultAnalyzerBatched<'a> {
    /// Creates a new analyzer for the given circuit.
    #[must_use]
    pub fn new(circuit: &'a TickCircuit) -> Self {
        let (gates, tick_gates, max_qubit) = Self::flatten_circuit(circuit);
        let locations = Self::extract_spacetime_locations(&gates);

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
            gates,
            tick_gates,
            max_qubit,
            locations,
            tick_locations,
        }
    }

    fn flatten_circuit(circuit: &TickCircuit) -> (Vec<AnalyzerGate>, Vec<Vec<usize>>, usize) {
        let mut gates = Vec::new();
        let mut tick_gates = vec![Vec::new(); circuit.num_ticks()];
        let mut max_qubit = 0usize;

        for (tick, gate) in circuit.iter_gate_batches_with_tick() {
            let idx = gates.len();
            let qubits = gate.qubits.to_vec();
            for qubit in &qubits {
                max_qubit = max_qubit.max(qubit.index());
            }
            if tick >= tick_gates.len() {
                tick_gates.resize_with(tick + 1, Vec::new);
            }
            tick_gates[tick].push(idx);
            gates.push(AnalyzerGate {
                tick,
                gate_type: gate.gate_type,
                qubits,
            });
        }

        (gates, tick_gates, max_qubit)
    }

    /// Extracts spacetime locations using raw index access.
    fn extract_spacetime_locations(gates: &[AnalyzerGate]) -> Vec<SpacetimeLocation> {
        let mut locations = Vec::new();

        for (idx, gate) in gates.iter().enumerate() {
            // Before location (fault before gate)
            locations.push(SpacetimeLocation {
                tick: gate.tick,
                qubits: gate.qubits.clone(),
                before: true,
                gate_type: gate.gate_type,
                gate_index: idx,
            });

            // After location for most gates (except prep which resets)
            if !matches!(gate.gate_type, GateType::PZ | GateType::QAlloc) {
                locations.push(SpacetimeLocation {
                    tick: gate.tick,
                    qubits: gate.qubits.clone(),
                    before: false,
                    gate_type: gate.gate_type,
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
        self.build_influence_map_with_tracked_paulis(&[])
    }

    /// Builds the fault influence map with tracked Pauli tracking.
    #[must_use]
    pub fn build_influence_map_with_tracked_paulis(
        &self,
        tracked_paulis: &[(&[usize], &[usize])],
    ) -> FaultInfluenceMap {
        let mut map = FaultInfluenceMap::new();

        // Extract all measurements from the circuit
        let measurements = self.extract_measurements();
        map.measurements.clone_from(&measurements);

        // Create simple detectors (one per measurement)
        for m in &measurements {
            map.detectors.push(DetectorId::single(*m));
        }

        // Create tracked-Pauli IDs
        for (i, _) in tracked_paulis.iter().enumerate() {
            map.tracked_paulis.push(TrackedPauliId {
                op_index: i,
                component: 0,
            });
        }

        // Initialize influence for each fault location
        for loc in &self.locations {
            map.influences
                .insert(loc.clone(), FaultInfluence::default());
        }

        // Create work buffers
        let max_qubit = self.max_qubit;
        let max_gate = self.gates.len();
        let mut buffers = AnalyzerWorkBuffers::new(max_qubit, max_gate);

        // Backward propagate from each measurement
        for measurement in &measurements {
            self.propagate_from_measurement_optimized(measurement, &mut map, &mut buffers);
        }

        // Backward propagate from each tracked Pauli
        for (i, (x_pos, z_pos)) in tracked_paulis.iter().enumerate() {
            let tracked_pauli_id = TrackedPauliId {
                op_index: i,
                component: 0,
            };
            self.propagate_from_tracked_pauli_optimized(
                x_pos,
                z_pos,
                &tracked_pauli_id,
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

        for gate in &self.gates {
            let basis = match gate.gate_type {
                GateType::MZ | GateType::MeasureFree => 0, // Z-basis
                _ => continue,
            };

            for qubit in &gate.qubits {
                measurements.push(MeasurementId {
                    tick: gate.tick,
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
        // Clear processed gates bitset for this tick
        buffers.gates_to_process.clear();

        // Get gates in this tick directly from pre-computed index
        let tick_gates = self
            .tick_gates
            .get(tick_idx)
            .map_or([].as_slice(), Vec::as_slice);

        // Find gates that touch active qubits
        for &gate_idx in tick_gates {
            let gate = &self.gates[gate_idx];

            // Check if any qubit is active
            let touches_active = gate.qubits.iter().any(|q| {
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
            let gate_idx = buffers.gates_to_process[i];
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
        let gate = &self.gates[idx];
        let gate_type = gate.gate_type;
        let qubits = gate.qubits.as_slice();

        match gate_type {
            GateType::CX if qubits.len() >= 2 => {
                for pair in qubits.chunks_exact(2) {
                    let control = pair[0].index();
                    let target = pair[1].index();

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
            }

            GateType::CZ if qubits.len() >= 2 => {
                for pair in qubits.chunks_exact(2) {
                    let q0 = pair[0].index();
                    let q1 = pair[1].index();

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
            }

            GateType::CY
            | GateType::SWAP
            | GateType::SXX
            | GateType::SXXdg
            | GateType::SYY
            | GateType::SYYdg
            | GateType::SZZ
            | GateType::SZZdg
                if qubits.len() >= 2 =>
            {
                for pair in qubits.chunks_exact(2) {
                    let q0 = pair[0];
                    let q1 = pair[1];
                    let pair = [(q0, q1)];
                    match gate_type {
                        GateType::CY => {
                            prop.cy(&pair);
                        }
                        GateType::SWAP => {
                            prop.swap(&pair);
                        }
                        GateType::SXX => {
                            prop.sxxdg(&pair);
                        }
                        GateType::SXXdg => {
                            prop.sxx(&pair);
                        }
                        GateType::SYY => {
                            prop.syydg(&pair);
                        }
                        GateType::SYYdg => {
                            prop.syy(&pair);
                        }
                        GateType::SZZ => {
                            prop.szzdg(&pair);
                        }
                        GateType::SZZdg => {
                            prop.szz(&pair);
                        }
                        _ => unreachable!(),
                    }
                    Self::update_active_qubit(q0.index(), prop, buffers);
                    Self::update_active_qubit(q1.index(), prop, buffers);
                }
            }

            GateType::H => {
                for qid in qubits {
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

            GateType::SX
            | GateType::SXdg
            | GateType::SY
            | GateType::SYdg
            | GateType::F
            | GateType::Fdg => {
                for qid in qubits {
                    let q = [QubitId(qid.index())];
                    match gate_type {
                        GateType::SX => {
                            prop.sxdg(&q);
                        }
                        GateType::SXdg => {
                            prop.sx(&q);
                        }
                        GateType::SY => {
                            prop.sydg(&q);
                        }
                        GateType::SYdg => {
                            prop.sy(&q);
                        }
                        GateType::F => {
                            prop.fdg(&q);
                        }
                        GateType::Fdg => {
                            prop.f(&q);
                        }
                        _ => unreachable!(),
                    }
                    Self::update_active_qubit(qid.index(), prop, buffers);
                }
            }

            GateType::SZ | GateType::SZdg => {
                for qid in qubits {
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

    /// Optimized backward propagation from a tracked Pauli.
    fn propagate_from_tracked_pauli_optimized(
        &self,
        x_positions: &[usize],
        z_positions: &[usize],
        tracked_pauli_id: &TrackedPauliId,
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
                Some(tracked_pauli_id),
                map,
                false,
            );

            self.apply_tick_backward_optimized(tick_idx, &mut prop, buffers);

            self.record_influences_at_tick_filtered(
                tick_idx,
                &prop,
                &dummy_detector,
                Some(tracked_pauli_id),
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
        tracked_pauli: Option<&TrackedPauliId>,
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
                        if let Some(op) = tracked_pauli {
                            influence.tracked_pauli_flips[1].push(*op);
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
                        if let Some(op) = tracked_pauli {
                            influence.tracked_pauli_flips[3].push(*op);
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

                    // Y fault: anticommutes with X or Z but NOT both (Y commutes with Y)
                    if obs_x ^ obs_z {
                        if let Some(op) = tracked_pauli {
                            influence.tracked_pauli_flips[2].push(*op);
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

            for (pauli, tracked_paulis) in influence.tracked_pauli_flips.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)] // Pauli index 0..2
                let pauli_u8 = pauli as u8;
                for tracked_pauli in tracked_paulis {
                    map.tracked_pauli_to_faults
                        .entry(*tracked_pauli)
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
    use pecos_quantum::TickCircuit;

    #[test]
    fn test_basic_analysis() {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1]);
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1)]);
        circuit.tick().mz(&[0, 1]);
        let analyzer = TickFaultAnalyzerBatched::new(&circuit);

        assert!(!analyzer.locations().is_empty());

        let map = analyzer.build_influence_map();

        assert!(!map.measurements.is_empty());
        assert!(!map.detectors.is_empty());
    }

    #[test]
    fn test_sparse_traversal() {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2, 3]);
        circuit.tick().h(&[0]).h(&[2]);
        circuit.tick().cx(&[(0, 1)]).cx(&[(2, 3)]);
        circuit.tick().mz(&[0, 1, 2, 3]);
        let analyzer = TickFaultAnalyzerBatched::new(&circuit);

        let map = analyzer.build_influence_map();

        assert_eq!(map.measurements.len(), 4);
        assert_eq!(map.detectors.len(), 4);
    }

    #[test]
    fn test_tracked_pauli_propagation() {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1]);
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1)]);
        circuit.tick().mz(&[0, 1]);
        let analyzer = TickFaultAnalyzerBatched::new(&circuit);

        let tracked_paulis = [(&[] as &[usize], &[1usize] as &[usize])];
        let map = analyzer.build_influence_map_with_tracked_paulis(&tracked_paulis);

        assert_eq!(map.tracked_paulis.len(), 1);
    }
}
