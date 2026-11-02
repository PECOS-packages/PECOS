// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Owned inspection snapshots linking SVG components to stored gate batches.

use crate::{Attribute, DagCircuit, DiagramRangeError, DiagramStyle, Gate, TickCircuit};
use std::collections::BTreeMap;
use std::ops::Range;

pub(crate) const OPERATION_ID_KEY: &str = "_pecos_inspection_operation";

/// Location in the unchanged source circuit from which a snapshot was made.
/// These are stored batch/node indices, not measurement IDs or gate instances.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InspectionSource {
    TickBatch { tick: usize, batch: usize },
    DagNode { node: usize },
}

/// One stored batch and its source metadata. Several SVG components may share
/// its ID, including the endpoints and connector of each batched interaction.
#[derive(Clone, Debug)]
pub struct InspectionOperation {
    pub id: String,
    pub source: InspectionSource,
    pub gate: Gate,
    pub metadata: BTreeMap<String, Attribute>,
}

/// Immutable, owned SVG and operation records. No source circuit is modified.
///
/// IDs are scoped to this source snapshot. Ranges of the same unchanged circuit
/// retain them, but independently edited, converted, or compacted circuits can
/// reuse a location for a different gate. Recorded trace origins in metadata
/// are separate from these presentation IDs.
pub struct CircuitInspection {
    svg: String,
    operations: Vec<InspectionOperation>,
}

impl CircuitInspection {
    #[must_use]
    pub fn svg(&self) -> &str {
        &self.svg
    }

    #[must_use]
    pub fn operations(&self) -> &[InspectionOperation] {
        &self.operations
    }
}

impl TickCircuit {
    /// Inspect every tick, retaining original batch locations and metadata.
    ///
    /// # Panics
    /// Panics if a tick is missing from the circuit's internal storage.
    #[must_use]
    pub fn inspect(&self, style: &DiagramStyle) -> CircuitInspection {
        let (tagged, operations) = self.inspection_parts(0..self.num_ticks());
        let svg = tagged.render_with(style).svg();
        CircuitInspection { svg, operations }
    }

    /// Inspect a half-open tick range. Original batch locations are not renumbered.
    ///
    /// # Errors
    /// Returns [`DiagramRangeError`] for an invalid range.
    ///
    /// # Panics
    /// Panics if a tick inside the validated range is missing from internal storage.
    pub fn inspect_range(
        &self,
        range: Range<usize>,
        style: &DiagramStyle,
    ) -> Result<CircuitInspection, DiagramRangeError> {
        crate::circuit_display::validate_diagram_range(&range, self.num_ticks())?;
        let (tagged, operations) = self.inspection_parts(range.clone());
        let svg = tagged.render_range(range, style)?.svg();
        Ok(CircuitInspection { svg, operations })
    }

    fn inspection_parts(&self, range: Range<usize>) -> (Self, Vec<InspectionOperation>) {
        let mut tagged = self.clone();
        let mut operations = Vec::new();
        for tick in range {
            for batch in self.get_tick(tick).unwrap().iter_gate_batches() {
                let index = batch.batch_index();
                let id = format!("tick:{tick}:batch:{index}");
                operations.push(InspectionOperation {
                    id: id.clone(),
                    source: InspectionSource::TickBatch { tick, batch: index },
                    gate: batch.as_gate().clone(),
                    metadata: batch.attrs().map(|(k, v)| (k.clone(), v.clone())).collect(),
                });
                tagged.get_tick_mut(tick).unwrap().set_gate_attr(
                    index,
                    OPERATION_ID_KEY,
                    Attribute::String(id),
                );
            }
        }
        (tagged, operations)
    }
}

impl DagCircuit {
    /// Inspect all dependency layers using original DAG node identities.
    #[must_use]
    pub fn inspect(&self, style: &DiagramStyle) -> CircuitInspection {
        let (tagged, operations) = self.inspection_parts(self.layers().flatten());
        let svg = tagged.render_with(style).svg();
        CircuitInspection { svg, operations }
    }

    /// Inspect a half-open dependency-layer range, retaining original node IDs.
    ///
    /// # Errors
    /// Returns [`DiagramRangeError`] for an invalid range.
    pub fn inspect_range(
        &self,
        range: Range<usize>,
        style: &DiagramStyle,
    ) -> Result<CircuitInspection, DiagramRangeError> {
        let layers: Vec<_> = self.layers().collect();
        crate::circuit_display::validate_diagram_range(&range, layers.len())?;
        let (tagged, operations) =
            self.inspection_parts(layers[range.clone()].iter().flatten().copied());
        let svg = tagged.render_range(range, style)?.svg();
        Ok(CircuitInspection { svg, operations })
    }

    fn inspection_parts(
        &self,
        nodes: impl Iterator<Item = usize>,
    ) -> (Self, Vec<InspectionOperation>) {
        let mut tagged = self.clone();
        let mut operations = Vec::new();
        for node in nodes {
            if let Some(gate) = self.gate(node) {
                let id = format!("node:{node}");
                operations.push(InspectionOperation {
                    id: id.clone(),
                    source: InspectionSource::DagNode { node },
                    gate: gate.clone(),
                    metadata: self.gate_attrs(node).cloned().unwrap_or_default(),
                });
                tagged.set_gate_attr(node, OPERATION_ID_KEY, Attribute::String(id));
            }
        }
        (tagged, operations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_sources_survive_range_selection_without_mutation() {
        let mut circuit = TickCircuit::new();
        circuit.tick().h(&[0, 1]);
        circuit
            .tick()
            .cx(&[(1, 0)])
            .meta("calibration", Attribute::String("example".into()));
        circuit.tick().mz(&[0, 1]);
        let style = DiagramStyle::builder().show_measurement_ids(true).build();
        let before = format!("{circuit:?}");
        let full = circuit.inspect(&style);
        let selected = circuit.inspect_range(1..2, &style).unwrap();
        assert_eq!(format!("{circuit:?}"), before);
        assert_eq!(selected.operations().len(), 1);
        let operation = &selected.operations()[0];
        assert_eq!(operation.id, "tick:1:batch:0");
        assert_eq!(
            operation.source,
            InspectionSource::TickBatch { tick: 1, batch: 0 }
        );
        assert_eq!(full.operations()[1].id, operation.id);
        assert_eq!(
            operation.gate.qubits.as_slice(),
            &[crate::QubitId(1), crate::QubitId(0)]
        );
        assert_eq!(
            operation.metadata["calibration"],
            Attribute::String("example".into())
        );
        assert!(!circuit.to_svg().contains("data-pecos-operation"));
        // Removing annotation groups must recover exactly the ordinary layout.
        let untagged = selected
            .svg()
            .lines()
            .filter(|line| !line.starts_with("<g data-pecos-operation=") && *line != "</g>")
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        assert_eq!(untagged, circuit.render_range(1..2, &style).unwrap().svg());
        circuit.tick().x(&[5]);
        assert!(!full.svg().contains("q5"));
    }

    #[test]
    fn expanded_batches_keep_one_source_for_all_endpoints_and_connectors() {
        let mut circuit = TickCircuit::new();
        circuit.tick().cx(&[(0, 2), (1, 3)]);
        let snapshot = circuit.inspect(&DiagramStyle::default());
        assert_eq!(snapshot.operations().len(), 1);
        // Two instances, each with two endpoints and one connector. No source
        // groups for intermediate wires, tick shading, or labels.
        assert_eq!(
            snapshot
                .svg()
                .matches("data-pecos-operation=\"tick:0:batch:0\"")
                .count(),
            6
        );
    }

    #[test]
    fn dag_sources_are_original_node_indices_not_layer_positions() {
        let mut circuit = DagCircuit::new();
        circuit.h(&[0]);
        circuit.x(&[1]);
        circuit.cx(&[(0, 1)]);
        let style = DiagramStyle::default();
        let full = circuit.inspect(&style);
        let selected = circuit.inspect_range(1..2, &style).unwrap();
        assert_eq!(selected.operations().len(), 1);
        let operation = &selected.operations()[0];
        let InspectionSource::DagNode { node } = operation.source else {
            panic!("expected DAG source")
        };
        assert_eq!(circuit.gate(node).unwrap().gate_type, crate::GateType::CX);
        assert!(full.operations().iter().any(|op| op.id == operation.id));
        assert_eq!(selected.svg().matches("data-pecos-operation=").count(), 3);
        assert!(
            circuit
                .inspect_range(2..2, &style)
                .unwrap()
                .operations()
                .is_empty()
        );
        assert!(circuit.inspect_range(2..3, &style).is_err());
    }
}
