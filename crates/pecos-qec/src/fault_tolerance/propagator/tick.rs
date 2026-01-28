// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Tick-based fault analysis for quantum circuits.
//!
//! This module provides [`TickFaultAnalyzer`] for analyzing fault tolerance properties
//! of circuits represented as [`TickCircuit`](pecos_quantum::TickCircuit).
//!
//! For better performance, consider using [`DagFaultAnalyzer`](super::DagFaultAnalyzer)
//! with DAG circuits, which provides 5-50x speedup through sparse traversal.

use super::types::{DetectorId, FaultInfluence, FaultInfluenceMap, LogicalId, MeasurementId};
use super::{SpacetimeLocation, extract_spacetime_locations};
use pecos_core::gate_type::GateType;
use pecos_qsim::PauliProp;
use pecos_quantum::TickCircuit;

// ============================================================================
// Tick-based Fault Analyzer
// ============================================================================

/// Propagates Paulis backward through a circuit to build influence maps.
///
/// The analyzer uses sparse traversal that only applies gate transformations to gates
/// touching qubits with non-trivial Paulis, providing speedup for circuits with local
/// connectivity (like surface codes).
///
/// For better performance on large circuits, consider using [`DagFaultAnalyzer`](super::DagFaultAnalyzer)
/// which provides 5-50x speedup through true sparse DAG traversal.
pub struct TickFaultAnalyzer<'a> {
    circuit: &'a TickCircuit,
    /// Fault locations extracted from the circuit.
    locations: Vec<SpacetimeLocation>,
    /// Pre-computed index: tick -> (`location_index`, before) pairs for O(1) lookup
    tick_locations: Vec<Vec<(usize, bool)>>,
    /// Maximum qubit index in the circuit.
    max_qubit: usize,
}

impl<'a> TickFaultAnalyzer<'a> {
    /// Creates a new backward propagator for the given circuit.
    #[must_use]
    pub fn new(circuit: &'a TickCircuit) -> Self {
        let locations = extract_spacetime_locations(circuit, false);

        // Build tick index for O(1) lookup
        let num_ticks = circuit.ticks().len();
        let mut tick_locations = vec![Vec::new(); num_ticks + 1];
        for (idx, loc) in locations.iter().enumerate() {
            if loc.tick < tick_locations.len() {
                tick_locations[loc.tick].push((idx, loc.before));
            }
        }

        // Find max qubit for active qubit tracking
        let mut max_qubit = 0;
        for tick in circuit.ticks() {
            for gate in tick.gates() {
                for qubit in &gate.qubits {
                    max_qubit = max_qubit.max(qubit.index());
                }
            }
        }

        Self {
            circuit,
            locations,
            tick_locations,
            max_qubit,
        }
    }

    /// Builds the complete fault influence map.
    ///
    /// This performs backward propagation from all measurements and
    /// creates a lookup table for fault classification.
    #[must_use]
    pub fn build_influence_map(&self) -> FaultInfluenceMap {
        self.build_influence_map_with_logicals(&[])
    }

    /// Builds the fault influence map with logical operator tracking.
    ///
    /// # Arguments
    ///
    /// * `logicals` - Logical operators as (`x_positions`, `z_positions`) pairs.
    ///   The first element of each pair is the X component positions,
    ///   the second is the Z component positions.
    #[must_use]
    pub fn build_influence_map_with_logicals(
        &self,
        logicals: &[(&[usize], &[usize])],
    ) -> FaultInfluenceMap {
        let mut map = FaultInfluenceMap::new();

        // Extract all measurements from the circuit
        let measurements = self.extract_measurements();
        map.measurements.clone_from(&measurements);

        // Create simple detectors (one per measurement for now)
        // TODO: Support comparison detectors for multi-round circuits
        for m in &measurements {
            map.detectors.push(DetectorId::single(*m));
        }

        // Create logical IDs
        for (i, _) in logicals.iter().enumerate() {
            map.logicals.push(LogicalId {
                logical_qubit: i,
                observable: 0, // Z observable
            });
        }

        // Initialize influence for each fault location
        for loc in &self.locations {
            map.influences
                .insert(loc.clone(), FaultInfluence::default());
        }

        // Backward propagate from each measurement
        for measurement in &measurements {
            self.propagate_from_measurement(measurement, &mut map);
        }

        // Backward propagate from each logical operator
        for (i, (x_pos, z_pos)) in logicals.iter().enumerate() {
            let logical_id = LogicalId {
                logical_qubit: i,
                observable: 0,
            };
            self.propagate_from_logical(x_pos, z_pos, &logical_id, &mut map);
        }

        // Build reverse maps
        self.build_reverse_maps(&mut map);

        map
    }

    /// Extracts all measurements from the circuit.
    fn extract_measurements(&self) -> Vec<MeasurementId> {
        let mut measurements = Vec::new();

        for (tick_idx, tick) in self.circuit.iter_ticks() {
            for gate in tick.gates() {
                // Currently only Z-basis measurements are supported
                let basis = match gate.gate_type {
                    GateType::Measure | GateType::MeasureFree => 0, // Z-basis
                    _ => continue,
                };

                for qubit in &gate.qubits {
                    measurements.push(MeasurementId {
                        tick: tick_idx,
                        qubit: qubit.index(),
                        basis,
                    });
                }
            }
        }

        measurements
    }

    /// Propagates backward from a measurement to find which faults flip it.
    ///
    /// We propagate the OBSERVABLE being measured backward through the circuit.
    /// An error P at location L flips the measurement if P anticommutes with
    /// the back-propagated observable at L.
    ///
    /// For Z-measurement, the observable is Z. Propagating Z backward tells us
    /// the effective observable at each location. An X error anticommutes with Z,
    /// so X errors flip the measurement where the observable has Z component.
    ///
    /// This uses sparse traversal: only gates touching qubits with non-trivial
    /// Paulis are processed, providing significant speedup for circuits with
    /// local connectivity.
    fn propagate_from_measurement(&self, measurement: &MeasurementId, map: &mut FaultInfluenceMap) {
        // Start with the observable being measured (not what flips it)
        // Z-measurement measures Z, X-measurement measures X
        let initial_pauli = if measurement.basis == 0 { 3u8 } else { 1u8 }; // Z or X

        let mut prop = PauliProp::new();
        match initial_pauli {
            1 => prop.add_x(measurement.qubit),
            3 => prop.add_z(measurement.qubit),
            _ => {}
        }

        let detector = DetectorId::single(*measurement);

        // Check faults at the measurement tick (before=true faults only)
        // These are right before the measurement and see the initial sensitivity
        self.record_influences_at_tick_filtered(
            measurement.tick,
            &prop,
            &detector,
            None,
            map,
            true, // only before=true locations
        );

        // Track active qubits for sparse traversal
        let mut active_qubits = vec![false; self.max_qubit + 1];
        if measurement.qubit <= self.max_qubit {
            active_qubits[measurement.qubit] = true;
        }

        // Propagate backward through ticks using sparse traversal
        // For each tick t, we need to handle before=false and before=true locations separately:
        // - before=false (after gates at t): check with sensitivity BEFORE applying gates backward
        // - before=true (before gates at t): check with sensitivity AFTER applying gates backward
        for tick_idx in (0..measurement.tick).rev() {
            // Check before=false locations (faults that happen after gates at this tick)
            // These see the sensitivity at the state "after tick t gates executed"
            self.record_influences_at_tick_filtered(
                tick_idx, &prop, &detector, None, map, false, // only before=false locations
            );

            // Apply gates at this tick backward - SPARSE: only gates touching active qubits
            if tick_idx < self.circuit.ticks().len() {
                let tick = &self.circuit.ticks()[tick_idx];
                for gate in tick.gates() {
                    // Check if this gate touches any active qubit
                    let touches_active = gate.qubits.iter().any(|q| {
                        let idx = q.index();
                        idx < active_qubits.len() && active_qubits[idx]
                    });

                    if touches_active {
                        // Apply gate backward
                        self.apply_gate_backward(&mut prop, gate);

                        // Update active qubits based on new Pauli state
                        for q in &gate.qubits {
                            let idx = q.index();
                            if idx < active_qubits.len() {
                                active_qubits[idx] = prop.contains_x(idx) || prop.contains_z(idx);
                            }
                        }
                    }
                }
            }

            // Check before=true locations (faults that happen before gates at this tick)
            // These see the sensitivity at the state "before tick t gates executed"
            self.record_influences_at_tick_filtered(
                tick_idx, &prop, &detector, None, map, true, // only before=true locations
            );
        }
    }

    /// Propagates backward from a logical operator.
    ///
    /// We propagate the logical OBSERVABLE backward through the circuit.
    /// An error P at location L flips the logical if P anticommutes with
    /// the back-propagated observable at L.
    ///
    /// This uses sparse traversal: only gates touching qubits with non-trivial
    /// Paulis are processed, providing significant speedup for circuits with
    /// local connectivity.
    fn propagate_from_logical(
        &self,
        x_positions: &[usize],
        z_positions: &[usize],
        logical_id: &LogicalId,
        map: &mut FaultInfluenceMap,
    ) {
        // Start with the logical observable itself (not swapped)
        // The recording function handles anticommutation checking
        let mut prop = PauliProp::new();

        // Track active qubits for sparse traversal
        let mut active_qubits = vec![false; self.max_qubit + 1];

        // X positions in logical -> X in prop
        for &q in x_positions {
            prop.add_x(q);
            if q <= self.max_qubit {
                active_qubits[q] = true;
            }
        }
        // Z positions in logical -> Z in prop
        for &q in z_positions {
            prop.add_z(q);
            if q <= self.max_qubit {
                active_qubits[q] = true;
            }
        }

        // Dummy detector for the recording function
        let dummy_detector = DetectorId::single(MeasurementId {
            tick: 0,
            qubit: 0,
            basis: 0,
        });

        // Propagate backward through all ticks using sparse traversal
        let num_ticks = self.circuit.ticks().len();
        for tick_idx in (0..num_ticks).rev() {
            // Check before=false locations (after gates at this tick)
            self.record_influences_at_tick_filtered(
                tick_idx,
                &prop,
                &dummy_detector,
                Some(logical_id),
                map,
                false,
            );

            // Apply gates backward - SPARSE: only gates touching active qubits
            let tick = &self.circuit.ticks()[tick_idx];
            for gate in tick.gates() {
                // Check if this gate touches any active qubit
                let touches_active = gate.qubits.iter().any(|q| {
                    let idx = q.index();
                    idx < active_qubits.len() && active_qubits[idx]
                });

                if touches_active {
                    // Apply gate backward
                    self.apply_gate_backward(&mut prop, gate);

                    // Update active qubits based on new Pauli state
                    for q in &gate.qubits {
                        let idx = q.index();
                        if idx < active_qubits.len() {
                            active_qubits[idx] = prop.contains_x(idx) || prop.contains_z(idx);
                        }
                    }
                }
            }

            // Check before=true locations (before gates at this tick)
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

    /// Records which fault locations at a tick would contribute to the propagated Pauli.
    /// Filters by before flag to handle timing correctly.
    ///
    /// The `prop` contains the back-propagated OBSERVABLE. A fault P anticommutes with
    /// the observable Q if they share positions where both are non-identity but different.
    ///
    /// Anticommutation rules (for single qubit):
    /// - X anticommutes with Z and Y
    /// - Z anticommutes with X and Y
    /// - Y anticommutes with X, Z, and Y
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
        // Use pre-computed tick index for O(1) lookup instead of O(n) linear scan
        if tick_idx >= self.tick_locations.len() {
            return;
        }

        for &(loc_idx, before) in &self.tick_locations[tick_idx] {
            if before != only_before {
                continue;
            }

            let loc = &self.locations[loc_idx];

            // Check each qubit in the fault location
            for (qubit_idx, qubit) in loc.qubits.iter().enumerate() {
                let q = qubit.index();

                // The back-propagated observable tells us what measurement is sensitive to
                // prop contains the observable, we check what anticommutes with it
                let obs_x = prop.contains_x(q);
                let obs_z = prop.contains_z(q);
                // Observable is: I (neither), X (x only), Z (z only), Y (both x and z)

                if let Some(influence) = map.influences.get_mut(loc) {
                    // X fault anticommutes with Z or Y observable
                    // (X anticommutes with Z, X anticommutes with Y=iXZ)
                    // X fault anticommutes with Z or Y observable
                    let x_flips = obs_z; // Z or Y (both have Z component)
                    if x_flips {
                        if let Some(log) = logical {
                            influence.logical_flips[1].push(*log);
                        } else {
                            influence.detector_flips[1].push(detector.clone());
                            influence.measurement_flips[1]
                                .extend(detector.measurements.iter().copied());
                            // Also record per-qubit influence for multi-qubit fault handling
                            influence
                                .per_qubit_detector_flips
                                .entry((qubit_idx, 1))
                                .or_default()
                                .push(detector.clone());
                        }
                    }

                    // Z fault anticommutes with X or Y observable
                    // (Z anticommutes with X, Z anticommutes with Y=iXZ)
                    let z_flips = obs_x; // X or Y (both have X component)
                    if z_flips {
                        if let Some(log) = logical {
                            influence.logical_flips[3].push(*log);
                        } else {
                            influence.detector_flips[3].push(detector.clone());
                            influence.measurement_flips[3]
                                .extend(detector.measurements.iter().copied());
                            // Also record per-qubit influence
                            influence
                                .per_qubit_detector_flips
                                .entry((qubit_idx, 3))
                                .or_default()
                                .push(detector.clone());
                        }
                    }

                    // Y fault = iXZ: Y anticommutes with X, Z, and Y
                    // Y anticommutes with observable if observable has X or Z component
                    let y_flips = obs_x || obs_z;
                    if y_flips {
                        if let Some(log) = logical {
                            influence.logical_flips[2].push(*log);
                        } else {
                            influence.detector_flips[2].push(detector.clone());
                            influence.measurement_flips[2]
                                .extend(detector.measurements.iter().copied());
                            // Also record per-qubit influence
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

    /// Applies a gate backward to a `PauliProp`.
    ///
    /// For Clifford gates, backward propagation follows specific rules:
    /// - CX: Same as forward (CX is self-adjoint)
    /// - H: Same as forward (H is self-adjoint)
    /// - SZ (S gate): X → -Y, Y → X, Z → Z (adjoint of forward)
    #[inline]
    fn apply_gate_backward(&self, prop: &mut PauliProp, gate: &pecos_core::Gate) {
        // Access gate.qubits directly - no allocation needed
        let qubits = &gate.qubits;

        match gate.gate_type {
            GateType::CX => {
                // CX is self-adjoint, same propagation as forward
                // X on control -> X on control AND target
                // X on target -> X on target
                // Z on control -> Z on control
                // Z on target -> Z on control AND target
                if qubits.len() >= 2 {
                    let control = qubits[0].index();
                    let target = qubits[1].index();

                    let ctrl_x = prop.contains_x(control);
                    let tgt_z = prop.contains_z(target);

                    // X spreads from control to target
                    if ctrl_x {
                        prop.add_x(target);
                    }
                    // Z spreads from target to control
                    if tgt_z {
                        prop.add_z(control);
                    }
                }
            }

            GateType::CZ => {
                // CZ is self-adjoint
                // X on either qubit -> X on that qubit AND Z on the other
                if qubits.len() >= 2 {
                    let q0 = qubits[0].index();
                    let q1 = qubits[1].index();

                    let x0 = prop.contains_x(q0);
                    let x1 = prop.contains_x(q1);

                    if x0 {
                        prop.add_z(q1);
                    }
                    if x1 {
                        prop.add_z(q0);
                    }
                }
            }

            GateType::H => {
                // H is self-adjoint: X <-> Z
                if let Some(qid) = qubits.first() {
                    let q = qid.index();
                    let has_x = prop.contains_x(q);
                    let has_z = prop.contains_z(q);

                    // Swap X and Z using toggle
                    if has_x && !has_z {
                        // Remove X by toggling, add Z
                        prop.add_x(q); // toggles off
                        prop.add_z(q);
                    } else if has_z && !has_x {
                        // Remove Z by toggling, add X
                        prop.add_z(q); // toggles off
                        prop.add_x(q);
                    }
                    // If both or neither, no change needed
                }
            }

            GateType::SZ => {
                // SZ† (adjoint): X -> -Y (we track as XZ), Y -> X, Z -> Z
                // Since we track Paulis mod phase, X -> Y means X -> XZ
                if let Some(qid) = qubits.first() {
                    let q = qid.index();
                    let has_x = prop.contains_x(q);
                    let has_z = prop.contains_z(q);

                    if has_x && !has_z {
                        // X -> XZ (Y with phase)
                        prop.add_z(q);
                    } else if has_x && has_z {
                        // Y (XZ) -> X: remove Z by toggling
                        prop.add_z(q); // toggles off
                    }
                    // Z -> Z (no change)
                }
            }

            GateType::SZdg => {
                // SZdg = SZ†, so SZdg† = SZ
                // Forward SZ: X -> Y, Y -> -X, Z -> Z
                if let Some(qid) = qubits.first() {
                    let q = qid.index();
                    let has_x = prop.contains_x(q);
                    let has_z = prop.contains_z(q);

                    if has_x && !has_z {
                        // X -> XZ (Y)
                        prop.add_z(q);
                    } else if has_x && has_z {
                        // Y -> X: remove Z by toggling
                        prop.add_z(q); // toggles off
                    }
                }
            }

            GateType::Prep | GateType::QAlloc => {
                // Preparation resets the qubit - backward propagation stops here
                // Any Pauli on a prepared qubit doesn't propagate further back
                // Toggle off both X and Z if present
                for qid in qubits {
                    let q = qid.index();
                    if prop.contains_x(q) {
                        prop.add_x(q); // toggles off
                    }
                    if prop.contains_z(q) {
                        prop.add_z(q); // toggles off
                    }
                }
            }

            // Pauli gates (X,Y,Z), Measure, MeasureFree, and other gates - no Pauli frame change
            _ => {}
        }
    }

    /// Builds reverse maps (detector -> faults, logical -> faults).
    fn build_reverse_maps(&self, map: &mut FaultInfluenceMap) {
        for (loc, influence) in &map.influences {
            for (pauli, detectors) in influence.detector_flips.iter().enumerate() {
                for detector in detectors {
                    map.detector_to_faults
                        .entry(detector.clone())
                        .or_default()
                        .push((loc.clone(), pauli as u8));
                }
            }

            for (pauli, logicals) in influence.logical_flips.iter().enumerate() {
                for logical in logicals {
                    map.logical_to_faults
                        .entry(*logical)
                        .or_default()
                        .push((loc.clone(), pauli as u8));
                }
            }
        }
    }
}
