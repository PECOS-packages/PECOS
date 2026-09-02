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

//! Pauli-frame lookup support for sampling Pauli-twirl masks.
//!
//! Twirl sites are emitted as three positional tracked-Pauli annotations per
//! site: X, Y, and Z. The DEM sampler samples decoder-facing detector and
//! observable bits. This lookup adds the deterministic frame update induced by a
//! user-supplied Pauli mask by XOR-ing precomputed detector/observable rows into
//! sampled shots.

use super::dem_builder::record_offset_to_absolute_index;
use super::propagator::{Direction, apply_gate, is_supported_prep_gate};
use pecos_core::gate_type::GateType;
use pecos_core::{Pauli, PauliString};
use pecos_quantum::{AnnotationKind, DagCircuit};
use pecos_simulators::PauliProp;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

type MeasurementRecordMap = BTreeMap<usize, Vec<(usize, usize)>>;

/// Errors returned while building or applying a Pauli-frame lookup.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PauliFrameLookupError {
    /// A tracked-Pauli annotation has no `meta_node` set.
    #[error(
        "tracked-Pauli annotation is missing its meta_node; cannot determine spacetime position"
    )]
    MissingMetaNode,

    /// A tracked-Pauli annotation's `meta_node` does not point at a
    /// `TrackedPauliMeta` gate in the DAG.
    #[error(
        "tracked-Pauli annotation references DAG node {meta_node}, which is missing or not a TrackedPauliMeta gate"
    )]
    MetaNodeNotTrackedPauliMeta { meta_node: usize },

    /// A measurement gate has malformed measurement IDs.
    #[error("measurement node {node} has {meas_ids} measurement id(s) for {qubits} qubit(s)")]
    MalformedMeasurementIds {
        node: usize,
        meas_ids: usize,
        qubits: usize,
    },

    /// Detector/observable metadata references a measurement record outside the
    /// circuit's measurement range.
    #[error(
        "{kind} {output} references measurement record offset {record}, but the circuit has {num_measurements} measurement(s)"
    )]
    InvalidRecordOffset {
        kind: &'static str,
        output: usize,
        record: i32,
        num_measurements: usize,
    },

    /// Twirl mask composition requires X/Y/Z triples per site.
    #[error("tracked-Pauli count {num_tracked_paulis} is not divisible by 3")]
    NonTripletTrackedPaulis { num_tracked_paulis: usize },

    /// The flat mask buffer length does not match the supplied shape.
    #[error("pauli mask buffer has length {len}, expected {expected} for shape ({rows}, {cols})")]
    MaskLengthMismatch {
        len: usize,
        expected: usize,
        rows: usize,
        cols: usize,
    },

    /// The number of mask rows must match the number of sampled shots.
    #[error("pauli mask row count {mask_rows} does not match num_shots {num_shots}")]
    MaskShotMismatch { mask_rows: usize, num_shots: usize },

    /// The number of mask columns must match the number of Pauli-twirl sites.
    #[error("pauli mask column count {mask_cols} does not match num_pauli_sites {num_pauli_sites}")]
    MaskSiteMismatch {
        mask_cols: usize,
        num_pauli_sites: usize,
    },

    /// Mask values must use 0=I, 1=X, 2=Y, 3=Z.
    #[error("pauli mask value {value} at row {row}, column {col} is outside 0..=3")]
    InvalidMaskValue { row: usize, col: usize, value: u8 },

    /// A sampled output row does not match the lookup dimensions.
    #[error("{kind} row {row} has length {actual}, expected {expected}")]
    OutputWidthMismatch {
        kind: &'static str,
        row: usize,
        actual: usize,
        expected: usize,
    },
}

/// Deterministic lookup from tracked-Pauli mask values to detector/observable flips.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PauliFrameLookup {
    num_pauli_sites: usize,
    num_detectors: usize,
    num_observables: usize,
    detector_rows: Vec<Vec<u32>>,
    observable_rows: Vec<Vec<u32>>,
}

/// Per-shot detector rows paired with per-shot observable rows.
type DetectorObservableRows = (Vec<Vec<bool>>, Vec<Vec<bool>>);

impl PauliFrameLookup {
    /// Build a Pauli-frame lookup from a DAG circuit and record-based detector
    /// and observable definitions.
    ///
    /// The circuit must carry positional tracked-Pauli annotations. The tracked
    /// Paulis are interpreted in groups of three per site, ordered X, Y, Z by
    /// the surface-code emitter.
    ///
    /// # Errors
    ///
    /// Returns an error when tracked-Pauli metadata is malformed, when tracked
    /// annotations are not X/Y/Z triples, or when detector/observable record
    /// offsets reference missing measurements.
    pub fn from_circuit(
        dag: &DagCircuit,
        detector_records: &[Vec<i32>],
        observable_records: &[Vec<i32>],
    ) -> Result<Self, PauliFrameLookupError> {
        let tracked_annotations: Vec<&pecos_quantum::PauliAnnotation> = dag
            .annotations()
            .iter()
            .filter(|ann| matches!(ann.kind, AnnotationKind::TrackedPauli))
            .collect();
        let mut meta_nodes: Vec<usize> = dag
            .nodes()
            .into_iter()
            .filter(|&node| {
                dag.gate(node)
                    .is_some_and(|gate| gate.gate_type == GateType::TrackedPauliMeta)
            })
            .collect();
        meta_nodes.sort_unstable();

        if tracked_annotations.len() != meta_nodes.len() {
            return Err(PauliFrameLookupError::MissingMetaNode);
        }
        let tracked: Vec<(&pecos_quantum::PauliAnnotation, usize)> =
            tracked_annotations.into_iter().zip(meta_nodes).collect();
        if !tracked.len().is_multiple_of(3) {
            return Err(PauliFrameLookupError::NonTripletTrackedPaulis {
                num_tracked_paulis: tracked.len(),
            });
        }

        let topo_order = dag.topological_order();
        let topo_positions: BTreeMap<usize, usize> = topo_order
            .iter()
            .enumerate()
            .map(|(pos, &node)| (node, pos))
            .collect();
        let (measurement_records, num_measurements) = measurement_records_by_node(dag)?;
        let detectors_by_measurement =
            outputs_by_measurement(num_measurements, detector_records, "detector")?;
        let observables_by_measurement =
            outputs_by_measurement(num_measurements, observable_records, "observable")?;

        let mut detector_rows = Vec::with_capacity(tracked.len());
        let mut observable_rows = Vec::with_capacity(tracked.len());

        for (ann, meta_node) in &tracked {
            if dag
                .gate(*meta_node)
                .is_none_or(|gate| gate.gate_type != GateType::TrackedPauliMeta)
            {
                return Err(PauliFrameLookupError::MetaNodeNotTrackedPauliMeta {
                    meta_node: *meta_node,
                });
            }
            let start_pos = *topo_positions.get(meta_node).ok_or(
                PauliFrameLookupError::MetaNodeNotTrackedPauliMeta {
                    meta_node: *meta_node,
                },
            )?;
            let affected_measurements = propagate_tracked_pauli_forward(
                dag,
                &topo_order,
                &measurement_records,
                start_pos,
                &ann.pauli,
            );
            detector_rows.push(measurements_to_output_row(
                &affected_measurements,
                &detectors_by_measurement,
            ));
            observable_rows.push(measurements_to_output_row(
                &affected_measurements,
                &observables_by_measurement,
            ));
        }

        Ok(Self {
            num_pauli_sites: tracked.len() / 3,
            num_detectors: detector_records.len(),
            num_observables: observable_records.len(),
            detector_rows,
            observable_rows,
        })
    }

    /// Number of mask sites. Each site has three tracked rows: X, Y, Z.
    #[must_use]
    pub fn num_pauli_sites(&self) -> usize {
        self.num_pauli_sites
    }

    /// Number of tracked-Pauli rows in the lookup.
    #[must_use]
    pub fn num_tracked_paulis(&self) -> usize {
        self.detector_rows.len()
    }

    /// Number of detector output columns.
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Number of observable output columns.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Return the detector and observable row for one tracked-Pauli index.
    #[must_use]
    pub fn row_effects(&self, tracked_idx: usize) -> Option<(&[u32], &[u32])> {
        self.detector_rows
            .get(tracked_idx)
            .zip(self.observable_rows.get(tracked_idx))
            .map(|(det, obs)| (det.as_slice(), obs.as_slice()))
    }

    /// Convert a flat `(num_shots, num_pauli_sites)` mask buffer to tracked-row firings.
    ///
    /// # Errors
    ///
    /// Returns an error when the mask shape does not match the lookup or when
    /// any mask value is outside `0..=3`.
    pub fn mask_firings(
        &self,
        masks: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<Vec<bool>>, PauliFrameLookupError> {
        self.validate_mask_shape(masks, rows, cols, rows)?;
        let mut firings = vec![vec![false; self.num_tracked_paulis()]; rows];
        for row in 0..rows {
            for col in 0..cols {
                let value = masks[row * cols + col];
                if value != 0 {
                    firings[row][mask_value_to_tracked_idx(col, value)] = true;
                }
            }
        }
        Ok(firings)
    }

    /// Compute the mask-induced XOR pattern for detectors and observables.
    ///
    /// Returns `(det_xor, obs_xor)` where `det_xor[i]` is the detector XOR
    /// pattern for shot `i` and `obs_xor[i]` is the observable XOR pattern.
    ///
    /// # Errors
    ///
    /// Returns an error when the mask shape does not match the lookup or when
    /// any mask value is outside `0..=3`.
    pub fn compute_mask_xor(
        &self,
        masks: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<DetectorObservableRows, PauliFrameLookupError> {
        let mut det_xor = vec![vec![false; self.num_detectors]; rows];
        let mut obs_xor = vec![vec![false; self.num_observables]; rows];
        self.apply_mask_values(masks, rows, cols, &mut det_xor, &mut obs_xor)?;
        Ok((det_xor, obs_xor))
    }

    /// XOR mask-induced frame flips into sampled detector and observable rows.
    ///
    /// # Errors
    ///
    /// Returns an error when the mask shape does not match the sampled batch,
    /// when any mask value is outside `0..=3`, or when sampled output row widths
    /// do not match the lookup dimensions.
    pub fn apply_mask_values(
        &self,
        masks: &[u8],
        rows: usize,
        cols: usize,
        det_events: &mut [Vec<bool>],
        obs_flips: &mut [Vec<bool>],
    ) -> Result<(), PauliFrameLookupError> {
        self.validate_mask_shape(masks, rows, cols, det_events.len())?;
        if obs_flips.len() != rows {
            return Err(PauliFrameLookupError::MaskShotMismatch {
                mask_rows: rows,
                num_shots: obs_flips.len(),
            });
        }

        for row in 0..rows {
            if det_events[row].len() != self.num_detectors {
                return Err(PauliFrameLookupError::OutputWidthMismatch {
                    kind: "detector",
                    row,
                    actual: det_events[row].len(),
                    expected: self.num_detectors,
                });
            }
            if obs_flips[row].len() != self.num_observables {
                return Err(PauliFrameLookupError::OutputWidthMismatch {
                    kind: "observable",
                    row,
                    actual: obs_flips[row].len(),
                    expected: self.num_observables,
                });
            }

            for col in 0..cols {
                let value = masks[row * cols + col];
                if value == 0 {
                    continue;
                }
                let tracked_idx = mask_value_to_tracked_idx(col, value);
                xor_row(&mut det_events[row], &self.detector_rows[tracked_idx]);
                xor_row(&mut obs_flips[row], &self.observable_rows[tracked_idx]);
            }
        }

        Ok(())
    }

    fn validate_mask_shape(
        &self,
        masks: &[u8],
        rows: usize,
        cols: usize,
        num_shots: usize,
    ) -> Result<(), PauliFrameLookupError> {
        let expected = rows.saturating_mul(cols);
        if masks.len() != expected {
            return Err(PauliFrameLookupError::MaskLengthMismatch {
                len: masks.len(),
                expected,
                rows,
                cols,
            });
        }
        if rows != num_shots {
            return Err(PauliFrameLookupError::MaskShotMismatch {
                mask_rows: rows,
                num_shots,
            });
        }
        if cols != self.num_pauli_sites {
            return Err(PauliFrameLookupError::MaskSiteMismatch {
                mask_cols: cols,
                num_pauli_sites: self.num_pauli_sites,
            });
        }
        for row in 0..rows {
            for col in 0..cols {
                let value = masks[row * cols + col];
                if value > 3 {
                    return Err(PauliFrameLookupError::InvalidMaskValue { row, col, value });
                }
            }
        }
        Ok(())
    }
}

fn mask_value_to_tracked_idx(site_idx: usize, value: u8) -> usize {
    site_idx * 3 + usize::from(value - 1)
}

fn xor_row(row: &mut [bool], indices: &[u32]) {
    for &idx in indices {
        if let Some(bit) = row.get_mut(idx as usize) {
            *bit = !*bit;
        }
    }
}

fn measurement_records_by_node(
    dag: &DagCircuit,
) -> Result<(MeasurementRecordMap, usize), PauliFrameLookupError> {
    let mut by_node = BTreeMap::new();
    let mut next_record = 0usize;
    let mut num_measurements = 0usize;

    for node in dag.topological_order() {
        let Some(gate) = dag.gate(node) else {
            continue;
        };
        // `MeasureLeaked` consumes no measurement record. Including it here
        // numbered it positionally while real measurements were numbered by
        // their `MeasId`, so a leaked measurement and a real one could claim the
        // same record.
        if !gate.gate_type.consumes_measurement_record() {
            continue;
        }
        if !gate.meas_ids.is_empty() && gate.meas_ids.len() != gate.qubits.len() {
            return Err(PauliFrameLookupError::MalformedMeasurementIds {
                node,
                meas_ids: gate.meas_ids.len(),
                qubits: gate.qubits.len(),
            });
        }

        let mut entries = Vec::with_capacity(gate.qubits.len());
        for (idx, qubit) in gate.qubits.iter().enumerate() {
            let record = if gate.meas_ids.is_empty() {
                let record = next_record;
                next_record += 1;
                record
            } else {
                gate.meas_ids[idx].index()
            };
            num_measurements = num_measurements.max(record + 1);
            entries.push((qubit.index(), record));
        }
        by_node.insert(node, entries);
    }

    Ok((by_node, num_measurements.max(next_record)))
}

fn outputs_by_measurement(
    num_measurements: usize,
    records_by_output: &[Vec<i32>],
    kind: &'static str,
) -> Result<Vec<Vec<usize>>, PauliFrameLookupError> {
    let mut outputs = vec![Vec::new(); num_measurements];
    for (output, records) in records_by_output.iter().enumerate() {
        for &record in records {
            let Some(measurement) = record_offset_to_absolute_index(num_measurements, record)
            else {
                return Err(PauliFrameLookupError::InvalidRecordOffset {
                    kind,
                    output,
                    record,
                    num_measurements,
                });
            };
            if measurement >= num_measurements {
                return Err(PauliFrameLookupError::InvalidRecordOffset {
                    kind,
                    output,
                    record,
                    num_measurements,
                });
            }
            outputs[measurement].push(output);
        }
    }
    Ok(outputs)
}

fn propagate_tracked_pauli_forward(
    dag: &DagCircuit,
    topo_order: &[usize],
    measurement_records: &BTreeMap<usize, Vec<(usize, usize)>>,
    start_pos: usize,
    pauli: &PauliString,
) -> BTreeSet<usize> {
    let mut prop = pauli_prop_from_string(pauli);
    let mut affected_measurements = BTreeSet::new();

    for &node in topo_order.iter().skip(start_pos + 1) {
        let Some(gate) = dag.gate(node) else {
            continue;
        };
        match gate.gate_type {
            GateType::TrackedPauliMeta => {}
            GateType::MX
            | GateType::MZ
            | GateType::MeasureFree
            | GateType::MeasureLeaked
            | GateType::MPZ => {
                if let Some(entries) = measurement_records.get(&node) {
                    for &(qubit, record) in entries {
                        let flips = if gate.gate_type == GateType::MX {
                            prop.contains_z(qubit)
                        } else {
                            prop.contains_x(qubit)
                        };
                        if flips {
                            affected_measurements.insert(record);
                        }
                    }
                }
                // Collapse, not reset: a non-destructive measurement absorbs
                // only the Z component -- the X component keeps flipping later
                // measurements on the same qubit. Only a discarded qubit
                // (`MeasureFree`) clears fully.
                for qubit in &gate.qubits {
                    crate::fault_tolerance::propagator::cross_measurement(
                        &mut prop,
                        qubit.index(),
                        gate.gate_type,
                        Direction::Forward,
                    );
                }
            }
            gate_type if is_supported_prep_gate(gate_type) => {
                for qubit in &gate.qubits {
                    clear_qubit(&mut prop, qubit.index());
                }
            }
            _ => {
                // Pauli-frame lookup intentionally preserves its historical
                // permissive treatment of unsupported gates.
                let _outcome = apply_gate(&mut prop, gate, Direction::Forward);
            }
        }
    }

    affected_measurements
}

fn measurements_to_output_row(
    measurements: &BTreeSet<usize>,
    outputs_by_measurement: &[Vec<usize>],
) -> Vec<u32> {
    let mut outputs = BTreeSet::new();
    for &measurement in measurements {
        if let Some(row) = outputs_by_measurement.get(measurement) {
            for &output in row {
                if !outputs.remove(&output) {
                    outputs.insert(output);
                }
            }
        }
    }
    outputs
        .into_iter()
        .map(|idx| u32::try_from(idx).expect("detector/observable index must fit into u32"))
        .collect()
}

fn pauli_prop_from_string(pauli: &PauliString) -> PauliProp {
    let mut prop = PauliProp::new();
    for (pauli, qubit) in pauli.iter_pairs() {
        let qubit = qubit.index();
        match pauli {
            Pauli::I => {}
            Pauli::X => prop.track_x(&[qubit]),
            Pauli::Z => prop.track_z(&[qubit]),
            Pauli::Y => prop.track_y(&[qubit]),
        }
    }
    prop
}

fn clear_qubit(prop: &mut PauliProp, qubit: usize) {
    if prop.contains_x(qubit) {
        prop.track_x(&[qubit]);
    }
    if prop.contains_z(qubit) {
        prop.track_z(&[qubit]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_quantum::Gate;

    /// A `MeasureLeaked` must affect a propagating Pauli exactly as an `MZ`
    /// does: both are non-destructive Z-collapses, executed identically by the
    /// simulators.
    #[test]
    fn measure_leaked_affects_a_propagating_pauli_like_an_mz() {
        fn later_measurement_flipped(leading: GateType) -> bool {
            let mut dag = DagCircuit::new();
            dag.pz(&[0]);
            let start = dag.add_gate_auto_wire(Gate::simple(
                GateType::TrackedPauliMeta,
                vec![pecos_quantum::QubitId::from(0usize)],
            ));
            let leading_gate = match leading {
                GateType::MeasureLeaked => Gate::measure_leaked(&[0usize]),
                _ => Gate::mz(&[0usize]),
            };
            dag.add_gate_auto_wire(leading_gate);
            let later = dag.add_gate_auto_wire(Gate::mz(&[0usize]));

            let topo_order = dag.topological_order();
            let start_pos = topo_order
                .iter()
                .position(|&n| n == start)
                .expect("meta node is in the order");
            let (records, _) = measurement_records_by_node(&dag).expect("mapping succeeds");
            let affected = propagate_tracked_pauli_forward(
                &dag,
                &topo_order,
                &records,
                start_pos,
                &PauliString::xs(&[0usize]),
            );
            // Only the *later* measurement matters: with a leading MZ the first
            // one is legitimately flipped, so a bare "anything affected" check
            // would compare different things.
            let later_record = records
                .get(&later)
                .and_then(|entries| entries.first())
                .map(|&(_, record)| record)
                .expect("the later MZ holds a record");
            affected.contains(&later_record)
        }

        assert_eq!(
            later_measurement_flipped(GateType::MeasureLeaked),
            later_measurement_flipped(GateType::MZ),
            "the two non-destructive measurements must treat a Pauli identically"
        );
        // And the absolute: collapse projects, it does not reset, so the X
        // survives the first measurement and flips the later one.
        assert!(
            later_measurement_flipped(GateType::MZ),
            "an X before a non-destructive MZ keeps flipping later measurements"
        );
    }

    /// `MeasureFree` does consume a record, so it must be numbered here.
    #[test]
    fn measure_free_claims_a_measurement_record() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0, 1]);
        let freed = dag.add_gate_auto_wire(Gate::mz_free(&[0usize]));
        let measured = dag.add_gate_auto_wire(Gate::mz(&[1usize]));

        let (by_node, num_measurements) =
            measurement_records_by_node(&dag).expect("mapping succeeds");

        assert_eq!(
            by_node.get(&freed).map(Vec::as_slice),
            Some([(0usize, 0usize)].as_slice())
        );
        assert_eq!(
            by_node.get(&measured).map(Vec::as_slice),
            Some([(1usize, 1usize)].as_slice())
        );
        assert_eq!(num_measurements, 2);
    }

    /// `MeasureLeaked` must not consume a measurement record here.
    ///
    /// It used to, and because `DagCircuit` mints no id for it, it took the
    /// positional branch and claimed record 0 -- the same record the first real
    /// measurement holds by its `MeasId`. Two different measurements then mapped
    /// to one record.
    #[test]
    fn measure_leaked_does_not_claim_a_measurement_record() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0, 1]);
        let leaked = dag.add_gate_auto_wire(Gate::measure_leaked(&[0usize]));
        let measured = dag.add_gate_auto_wire(Gate::mz(&[1usize]));

        let (by_node, num_measurements) =
            measurement_records_by_node(&dag).expect("mapping succeeds");

        assert!(
            !by_node.contains_key(&leaked),
            "a leaked measurement holds no record"
        );
        assert_eq!(
            by_node.get(&measured).map(Vec::as_slice),
            Some([(1usize, 0usize)].as_slice()),
            "the real measurement keeps record 0"
        );
        assert_eq!(num_measurements, 1);
    }
}
