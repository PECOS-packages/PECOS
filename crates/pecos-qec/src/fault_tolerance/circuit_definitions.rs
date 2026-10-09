// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

//! Circuit detector and observable definitions in measurement emission order.

use super::dem_builder::{
    DemBuilderError, parse_detectors_json, parse_observables_json, record_offset_to_absolute_index,
};
use pecos_core::{MeasId, PauliString};
use pecos_quantum::{AnnotationKind, Attribute, DagCircuit, PauliAnnotation, TickCircuit};
use std::collections::BTreeMap;

/// The kind of a measurement-readout annotation.
#[derive(Debug, Clone, PartialEq)]
pub enum DefinitionKind {
    /// A detector, with optional coordinates (empty means absent).
    Detector { coords: Vec<f64> },
    /// A logical observable.
    Observable,
}

/// Circuit-independent annotation input; a Pauli string is not required.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnotationDefinition {
    /// Detector or observable role.
    pub kind: DefinitionKind,
    /// Stable identities, in annotation order, with duplicates retained.
    pub measurement_ids: Vec<MeasId>,
    /// Optional human-readable name.
    pub label: Option<String>,
    /// Optional Pauli string supplied by the annotation source.
    pub pauli: Option<PauliString>,
}

impl AnnotationDefinition {
    /// Converts a readout annotation; tracked Paulis return `None`.
    #[must_use]
    pub fn from_annotation(annotation: &PauliAnnotation) -> Option<Self> {
        let (kind, measurement_ids) = match &annotation.kind {
            AnnotationKind::Detector {
                measurement_ids,
                coords,
            } => (
                DefinitionKind::Detector {
                    coords: coords.clone(),
                },
                measurement_ids,
            ),
            AnnotationKind::Observable { measurement_ids } => {
                (DefinitionKind::Observable, measurement_ids)
            }
            AnnotationKind::TrackedPauli => return None,
        };
        Some(Self {
            kind,
            measurement_ids: measurement_ids.clone(),
            label: annotation.label.clone(),
            pauli: Some(annotation.pauli.clone()),
        })
    }
}

/// Resolved detector; repeated positions retain their multiplicity.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedDetector {
    /// Declared metadata id or index among annotations of this kind.
    pub id: u32,
    /// Metadata coordinates, otherwise nonempty annotation coordinates.
    pub coords: Option<Vec<f64>>,
    /// Optional human-readable name.
    pub label: Option<String>,
    /// Absolute emission positions, preserving source order and multiplicity.
    pub measurements: Vec<usize>,
}

/// Resolved observable; metadata alone does not supply a Pauli string.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedObservable {
    /// Declared metadata id or index among annotations of this kind.
    pub id: u32,
    /// Optional human-readable name.
    pub label: Option<String>,
    /// Optional Pauli string supplied by the annotation source.
    pub pauli: Option<PauliString>,
    /// Absolute emission positions, preserving source order and multiplicity.
    pub measurements: Vec<usize>,
}

/// Definitions sorted by id, with positions indexing measurement emission order.
#[derive(Debug, Clone, PartialEq)]
pub struct CircuitDefinitions {
    /// Detectors sorted by id.
    pub detectors: Vec<ResolvedDetector>,
    /// Observables sorted by id.
    pub observables: Vec<ResolvedObservable>,
    /// Number of emitted measurement records, including records without ids.
    pub num_measurements: usize,
}

/// Invalid circuit definitions or measurement identity/order data.
#[derive(Debug, Clone, thiserror::Error)]
pub enum DefinitionError {
    #[error("detector id {id}: coordinate length {length} must be zero or three for a DEM")]
    InvalidDetectorCoordinates { id: u32, length: usize },
    #[error("{kind} metadata: {source}")]
    Parse {
        kind: &'static str,
        #[source]
        source: DemBuilderError,
    },
    #[error("{kind} id {id} is repeated in metadata")]
    DuplicateMetadataId { kind: &'static str, id: u32 },
    #[error(
        "{kind} id {id}: record offset {record} is out of range for a circuit with {num_measurements} measurement(s)"
    )]
    NegativeRecordOutOfRange {
        kind: &'static str,
        id: u32,
        record: i32,
        num_measurements: usize,
    },
    #[error(
        "{kind} id {id}: record offset {record} is out of range for a circuit with {num_measurements} measurement(s)"
    )]
    AbsoluteRecordOutOfRange {
        kind: &'static str,
        id: u32,
        record: i32,
        num_measurements: usize,
    },
    #[error(
        "{kind} id {id}: meas_id {measurement_id} is not present in the circuit's measurements"
    )]
    UnknownMeasurementId {
        kind: &'static str,
        id: u32,
        measurement_id: usize,
    },
    #[error("measurement id {measurement_id} occurs at emission positions {first} and {second}")]
    DuplicateEmissionId {
        measurement_id: usize,
        first: usize,
        second: usize,
    },
    #[error("{kind} id {id}: records resolve to {records:?}, meas_ids to {meas_ids:?}")]
    ReferenceMismatch {
        kind: &'static str,
        id: u32,
        records: Vec<usize>,
        meas_ids: Vec<usize>,
    },
    #[error("{kind} annotation index {index} does not fit u32")]
    AnnotationIdOverflow { kind: &'static str, index: usize },
    #[error("{kind} ids differ: metadata {metadata:?}, annotations {annotations:?}")]
    IdSetMismatch {
        kind: &'static str,
        metadata: Vec<u32>,
        annotations: Vec<u32>,
    },
    #[error(
        "{kind} id {id}: metadata positions {metadata:?} differ from annotation positions {annotations:?}"
    )]
    SourceMismatch {
        kind: &'static str,
        id: u32,
        metadata: Vec<usize>,
        annotations: Vec<usize>,
    },
    #[error("{kind} id {id}: conflicting labels {metadata:?} and {annotation:?}")]
    LabelConflict {
        kind: &'static str,
        id: u32,
        metadata: String,
        annotation: String,
    },
    #[error("attribute {attribute:?} must be a string, got {value:?}")]
    NonStringAttribute {
        attribute: &'static str,
        value: Attribute,
    },
    #[error("num_measurements {value:?} does not parse as usize")]
    InvalidMeasurementCount { value: String },
    #[error(
        "num_measurements={declared} disagrees with the {actual} measurement(s) the circuit performs; the declared count must match so detector/observable record offsets resolve correctly"
    )]
    MeasurementCountMismatch { declared: usize, actual: usize },
}

struct Definition {
    measurements: Vec<usize>,
    coords: Option<Vec<f64>>,
    label: Option<String>,
    pauli: Option<PauliString>,
}

type Definitions = BTreeMap<u32, Definition>;

fn same_multiset(left: &[usize], right: &[usize]) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}

struct Resolver {
    positions: BTreeMap<usize, usize>,
    num_measurements: usize,
}

impl Resolver {
    fn ids(
        &self,
        kind: &'static str,
        id: u32,
        ids: impl IntoIterator<Item = usize>,
    ) -> Result<Vec<usize>, DefinitionError> {
        ids.into_iter()
            .map(|measurement_id| {
                self.positions.get(&measurement_id).copied().ok_or(
                    DefinitionError::UnknownMeasurementId {
                        kind,
                        id,
                        measurement_id,
                    },
                )
            })
            .collect()
    }

    fn references(
        &self,
        kind: &'static str,
        id: u32,
        records: &[i32],
        ids: &[usize],
    ) -> Result<Vec<usize>, DefinitionError> {
        let positions = records
            .iter()
            .map(|&record| {
                let position = record_offset_to_absolute_index(self.num_measurements, record);
                if record < 0 && position.is_none() {
                    return Err(DefinitionError::NegativeRecordOutOfRange {
                        kind,
                        id,
                        record,
                        num_measurements: self.num_measurements,
                    });
                }
                position
                    .filter(|&position| position < self.num_measurements)
                    .ok_or(DefinitionError::AbsoluteRecordOutOfRange {
                        kind,
                        id,
                        record,
                        num_measurements: self.num_measurements,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let id_positions = self.ids(kind, id, ids.iter().copied())?;
        if !records.is_empty() && !ids.is_empty() && !same_multiset(&positions, &id_positions) {
            return Err(DefinitionError::ReferenceMismatch {
                kind,
                id,
                records: positions,
                meas_ids: id_positions,
            });
        }
        Ok(if records.is_empty() {
            id_positions
        } else {
            positions
        })
    }
}

fn insert_metadata(
    definitions: &mut Definitions,
    kind: &'static str,
    id: u32,
    definition: Definition,
) -> Result<(), DefinitionError> {
    if definitions.insert(id, definition).is_some() {
        return Err(DefinitionError::DuplicateMetadataId { kind, id });
    }
    Ok(())
}

fn merge(
    kind: &'static str,
    mut metadata: Definitions,
    annotations: Definitions,
) -> Result<Definitions, DefinitionError> {
    if metadata.is_empty() {
        return Ok(annotations);
    }
    if annotations.is_empty() {
        return Ok(metadata);
    }
    if !metadata.keys().eq(annotations.keys()) {
        return Err(DefinitionError::IdSetMismatch {
            kind,
            metadata: metadata.keys().copied().collect(),
            annotations: annotations.keys().copied().collect(),
        });
    }
    for (id, annotation) in annotations {
        let definition = metadata.get_mut(&id).expect("id sets were checked");
        if !same_multiset(&definition.measurements, &annotation.measurements) {
            return Err(DefinitionError::SourceMismatch {
                kind,
                id,
                metadata: definition.measurements.clone(),
                annotations: annotation.measurements,
            });
        }
        if let (Some(metadata), Some(annotation)) = (&definition.label, &annotation.label)
            && metadata != annotation
        {
            return Err(DefinitionError::LabelConflict {
                kind,
                id,
                metadata: metadata.clone(),
                annotation: annotation.clone(),
            });
        }
        definition.coords = definition.coords.take().or(annotation.coords);
        definition.label = definition.label.take().or(annotation.label);
        definition.pauli = annotation.pauli;
    }
    Ok(metadata)
}

/// Resolve metadata and annotations against an explicit measurement emission list.
///
/// # Errors
/// Returns an error for malformed metadata, ambiguous identities, invalid references,
/// or disagreement between definition sources. Empty metadata defines nothing.
pub fn resolve_definitions(
    detectors_json: Option<&str>,
    observables_json: Option<&str>,
    annotations: &[AnnotationDefinition],
    emission: &[Option<MeasId>],
) -> Result<CircuitDefinitions, DefinitionError> {
    let mut resolver = Resolver {
        positions: BTreeMap::new(),
        num_measurements: emission.len(),
    };
    for (second, measurement_id) in emission.iter().enumerate() {
        if let Some(measurement_id) = measurement_id {
            let measurement_id = measurement_id.index();
            if let Some(first) = resolver.positions.insert(measurement_id, second) {
                return Err(DefinitionError::DuplicateEmissionId {
                    measurement_id,
                    first,
                    second,
                });
            }
        }
    }
    let mut detectors = Definitions::new();
    for entry in detectors_json
        .map(parse_detectors_json)
        .transpose()
        .map_err(|source| DefinitionError::Parse {
            kind: "detector",
            source,
        })?
        .unwrap_or_default()
    {
        let definition = Definition {
            measurements: resolver.references(
                "detector",
                entry.id,
                &entry.records,
                &entry.meas_ids,
            )?,
            coords: entry.coords.map(Vec::from),
            label: entry.label,
            pauli: None,
        };
        insert_metadata(&mut detectors, "detector", entry.id, definition)?;
    }
    let mut observables = Definitions::new();
    for entry in observables_json
        .map(parse_observables_json)
        .transpose()
        .map_err(|source| DefinitionError::Parse {
            kind: "observable",
            source,
        })?
        .unwrap_or_default()
    {
        let definition = Definition {
            measurements: resolver.references(
                "observable",
                entry.id,
                &entry.records,
                &entry.meas_ids,
            )?,
            coords: None,
            label: entry.label,
            pauli: None,
        };
        insert_metadata(&mut observables, "observable", entry.id, definition)?;
    }
    let mut annotation_detectors = Definitions::new();
    let mut annotation_observables = Definitions::new();
    for annotation in annotations {
        let (kind, definitions, coords) = match &annotation.kind {
            DefinitionKind::Detector { coords } => (
                "detector",
                &mut annotation_detectors,
                (!coords.is_empty()).then(|| coords.clone()),
            ),
            DefinitionKind::Observable => ("observable", &mut annotation_observables, None),
        };
        let index = definitions.len();
        let id = u32::try_from(index)
            .map_err(|_| DefinitionError::AnnotationIdOverflow { kind, index })?;
        definitions.insert(
            id,
            Definition {
                measurements: resolver.ids(
                    kind,
                    id,
                    annotation.measurement_ids.iter().map(|id| id.index()),
                )?,
                coords,
                label: annotation.label.clone(),
                pauli: annotation.pauli.clone(),
            },
        );
    }
    Ok(CircuitDefinitions {
        detectors: merge("detector", detectors, annotation_detectors)?
            .into_iter()
            .map(|(id, d)| ResolvedDetector {
                id,
                coords: d.coords,
                label: d.label,
                measurements: d.measurements,
            })
            .collect(),
        observables: merge("observable", observables, annotation_observables)?
            .into_iter()
            .map(|(id, d)| ResolvedObservable {
                id,
                label: d.label,
                pauli: d.pauli,
                measurements: d.measurements,
            })
            .collect(),
        num_measurements: emission.len(),
    })
}

fn string_attribute<'a>(
    attribute: &'static str,
    value: Option<&'a Attribute>,
) -> Result<Option<&'a str>, DefinitionError> {
    match value {
        None => Ok(None),
        Some(Attribute::String(value)) => Ok(Some(value)),
        Some(value) => Err(DefinitionError::NonStringAttribute {
            attribute,
            value: value.clone(),
        }),
    }
}

/// Check a declared `num_measurements` against the emitted record count.
///
/// `None` means the circuit declares no count.
///
/// # Errors
/// Rejects a count that does not parse as `usize` or differs from `actual`.
pub fn check_measurement_count(
    declared: Option<&str>,
    actual: usize,
) -> Result<(), DefinitionError> {
    let Some(value) = declared else {
        return Ok(());
    };
    let declared =
        value
            .parse::<usize>()
            .map_err(|_| DefinitionError::InvalidMeasurementCount {
                value: value.to_owned(),
            })?;
    if declared != actual {
        return Err(DefinitionError::MeasurementCountMismatch { declared, actual });
    }
    Ok(())
}

fn resolve_circuit<'a>(
    get_attribute: impl Fn(&str) -> Option<&'a Attribute>,
    annotations: &[PauliAnnotation],
    emission: &[Option<MeasId>],
) -> Result<CircuitDefinitions, DefinitionError> {
    let detectors = string_attribute("detectors", get_attribute("detectors"))?;
    let observables = string_attribute("observables", get_attribute("observables"))?;
    check_measurement_count(
        string_attribute("num_measurements", get_attribute("num_measurements"))?,
        emission.len(),
    )?;
    let annotations: Vec<_> = annotations
        .iter()
        .filter_map(AnnotationDefinition::from_annotation)
        .collect();
    resolve_definitions(detectors, observables, &annotations, emission)
}

/// A tick circuit's measurement records in emission order: tick, batch storage
/// order, instance, then qubit-list order.
///
/// A batch's qubit list is its instances' qubit lists in instance order, and
/// `Gate::validate` holds every measurement gate to either no ids or one per
/// qubit, so walking each batch's qubits visits the records in emission order.
/// Entries are `None` for measurements that carry no id.
#[must_use]
pub fn tick_circuit_emission(circuit: &TickCircuit) -> Vec<Option<MeasId>> {
    let mut emission = Vec::new();
    for batch in circuit.iter_gate_batches() {
        if batch.gate_type.consumes_measurement_record() {
            emission
                .extend((0..batch.qubits.len()).map(|index| batch.meas_ids.get(index).copied()));
        }
    }
    emission
}

/// Return DAG nodes in emission order: topological order keyed by node index.
#[must_use]
pub fn dag_circuit_emission_order(circuit: &DagCircuit) -> Vec<usize> {
    circuit
        .as_dag()
        .lexicographical_topological_sort(|node| node)
}

/// A DAG's measurement records in emission order: topological order keyed by
/// node index, then qubit-list order.
///
/// For a DAG converted from a `TickCircuit` this is the tick circuit's order.
/// The unkeyed `DagCircuit::topological_order` is a depth-first order and can
/// put an independent later measurement first.
#[must_use]
pub fn dag_circuit_emission(circuit: &DagCircuit) -> Vec<Option<MeasId>> {
    let mut emission = Vec::new();
    for node in dag_circuit_emission_order(circuit) {
        let Some(gate) = circuit.gate(node) else {
            continue;
        };
        if gate.gate_type.consumes_measurement_record() {
            emission.extend((0..gate.qubits.len()).map(|index| gate.meas_ids.get(index).copied()));
        }
    }
    emission
}

/// Read a tick circuit's definitions against [`tick_circuit_emission`].
///
/// # Errors
/// Rejects invalid circuit definitions.
pub fn definitions_from_tick_circuit(
    circuit: &TickCircuit,
) -> Result<CircuitDefinitions, DefinitionError> {
    resolve_circuit(
        |key| circuit.get_meta(key),
        circuit.annotations(),
        &tick_circuit_emission(circuit),
    )
}

/// Read a DAG's definitions against [`dag_circuit_emission`].
///
/// # Errors
/// Rejects invalid circuit definitions.
pub fn definitions_from_dag_circuit(
    circuit: &DagCircuit,
) -> Result<CircuitDefinitions, DefinitionError> {
    resolve_circuit(
        |key| circuit.get_attr(key),
        circuit.annotations(),
        &dag_circuit_emission(circuit),
    )
}

#[cfg(test)]
mod tests;
