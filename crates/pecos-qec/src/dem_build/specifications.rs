// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Typed value specifications, resolved independently of program execution.

use super::bindings::{DemBindingError, MeasurementLedgerEntry, ResultTrace, ResultTraceIndex};
use super::{DemAnnotationOrigin, DemDefinitionOrigin, DemOutputSelector};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;

type Result<T> = std::result::Result<T, DemBindingError>;

/// A named scalar result selector. Python checks string tags and integer,
/// non-boolean, non-negative occurrences and elements before conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultRef {
    tag: String,
    occurrence: usize,
    element: Option<usize>,
}

impl ResultRef {
    /// # Errors
    /// Rejects empty tags.
    pub fn new(tag: impl Into<String>, occurrence: usize, element: Option<usize>) -> Result<Self> {
        let tag = tag.into();
        if tag.is_empty() {
            return Err(DemBindingError::value(
                "result_ref tag must be a non-empty string",
            ));
        }
        Ok(Self {
            tag,
            occurrence,
            element,
        })
    }

    #[must_use]
    pub fn tag(&self) -> &str {
        &self.tag
    }
    #[must_use]
    pub fn occurrence(&self) -> usize {
        self.occurrence
    }
    #[must_use]
    pub fn element(&self) -> Option<usize> {
        self.element
    }

    pub(super) fn indexed(tag: String, occurrence: usize, element: Option<usize>) -> Result<Self> {
        Self::new(tag, occurrence, element)
    }

    pub(super) fn to_json(&self) -> Value {
        json!({"tag": self.tag, "occurrence": self.occurrence, "element": self.element})
    }
}

/// A positive back-offset `k` for `rec[-k]`, or a named measurement.
/// Python checks record offsets are negative, non-boolean integers and converts
/// their magnitude before construction. It also converts bare string tags and
/// rejects unsupported reference objects. Bare tags use `MeasurementRef::tag`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeasurementRef {
    Record(NonZeroUsize),
    Result(ResultRef),
}

impl MeasurementRef {
    /// # Errors
    /// Rejects a zero back-offset.
    pub fn record(back_offset: usize) -> Result<Self> {
        NonZeroUsize::new(back_offset)
            .map(Self::Record)
            .ok_or_else(|| {
                DemBindingError::value("measurement record back-offset must be at least one")
            })
    }

    /// Bare-tag shorthand for occurrence zero with no element selector.
    /// # Errors
    /// Rejects an empty tag.
    pub fn tag(tag: impl Into<String>) -> Result<Self> {
        ResultRef::new(tag, 0, None).map(Self::Result)
    }
}

/// A reference-based source, or an output whose support is supplied by inference.
/// Python checks reference object types, string tags and non-boolean,
/// non-negative integer elements before conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DemSpecSource {
    References(Vec<MeasurementRef>),
    Inferred(DemOutputSelector),
}

impl DemSpecSource {
    /// # Errors
    /// Rejects empty tags.
    pub fn inferred(tag: impl Into<String>, element: usize) -> Result<Self> {
        let tag = tag.into();
        if tag.is_empty() {
            return Err(DemBindingError::value(
                "inferred output tag must be a non-empty string",
            ));
        }
        Ok(Self::Inferred(DemOutputSelector { tag, element }))
    }

    /// # Errors
    /// Rejects an empty reference list.
    pub fn references(refs: Vec<MeasurementRef>) -> Result<Self> {
        if refs.is_empty() {
            return Err(DemBindingError::value(
                "detectors and observables must reference at least one measurement",
            ));
        }
        Ok(Self::References(refs))
    }
}

/// Sibling of `DemDefinition` for optional IDs and typed references.
/// Python checks definition objects, non-boolean, non-negative integer IDs,
/// coordinate iterability/numeric conversion and metadata mapping types.
/// Coordinates retain arbitrary length. Python object conversion and metadata
/// JSON-serializability checks belong at the boundary; `serde_json::Value` cannot
/// contain non-finite numbers, including with `arbitrary_precision` enabled.
#[derive(Clone, Debug, PartialEq)]
pub struct DemSpec {
    pub id: Option<usize>,
    pub source: DemSpecSource,
    pub coords: Option<Vec<f64>>,
    pub metadata: Map<String, Value>,
}

/// A resolved parity with coordinates kept as floats rather than JSON numbers.
/// This preserves non-finite coordinates for the downstream coordinate validator.
/// Inputs have passed the Python boundary checks documented on `DemSpec`.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedDemSpec {
    pub id: usize,
    pub measurement_ids: Vec<usize>,
    pub coords: Option<Vec<f64>>,
    pub metadata: Map<String, Value>,
}

/// Resolved schema values, without fingerprinting or circuit preparation.
/// Inputs have passed the boundary checks on `DemSpec` and `ResultTrace`.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedDemSpecs {
    pub detectors: Vec<ResolvedDemSpec>,
    pub observables: Vec<ResolvedDemSpec>,
    pub ledger: Vec<MeasurementLedgerEntry>,
    pub result_ids_by_tag: BTreeMap<String, Vec<Option<usize>>>,
    pub definition_origins: Vec<DemDefinitionOrigin>,
}

/// Resolve definitions in Python check order: runtime identities, traces,
/// detectors and their density, then observables and their uniqueness.
/// Named references deliberately do not add a runtime-membership check; the
/// circuit-definition reader validates those identities downstream.
/// # Errors
/// Rejects invalid references, missing inferred supports, dead parities,
/// reserved metadata, and invalid definition IDs.
pub fn resolve_dem_specs(
    detectors: &[DemSpec],
    observables: &[DemSpec],
    runtime_order: &[usize],
    traces: &[ResultTrace],
    inferred_supports: &BTreeMap<(String, usize), Vec<usize>>,
) -> Result<ResolvedDemSpecs> {
    let index = ResultTraceIndex::new(runtime_order, traces)?;
    let mut origins = Vec::new();
    let detectors = resolve_role(
        "detector",
        detectors,
        &index,
        inferred_supports,
        &mut origins,
    )?;
    let observables = resolve_role(
        "observable",
        observables,
        &index,
        inferred_supports,
        &mut origins,
    )?;
    origins.sort_by_key(|origin| (origin.role != "detector", origin.id));
    Ok(ResolvedDemSpecs {
        detectors,
        observables,
        ledger: index.ledger(),
        result_ids_by_tag: index.named_bindings(None),
        definition_origins: origins,
    })
}

fn resolve_role(
    role: &'static str,
    specs: &[DemSpec],
    index: &ResultTraceIndex,
    supports: &BTreeMap<(String, usize), Vec<usize>>,
    origins: &mut Vec<DemDefinitionOrigin>,
) -> Result<Vec<ResolvedDemSpec>> {
    let mut entries = Vec::new();
    for (position, spec) in specs.iter().enumerate() {
        let id = spec.id.unwrap_or(position);
        let context = format!("{role} {id}");
        let (ids, origin) = match &spec.source {
            DemSpecSource::References(refs) => {
                // Enum variants are public, so enforce constructor invariants here too.
                DemSpecSource::references(refs.clone())?;
                (
                    refs.iter()
                        .map(|r| index.resolve_ref(r))
                        .collect::<Result<Vec<_>>>()?,
                    DemAnnotationOrigin::Explicit,
                )
            }
            DemSpecSource::Inferred(output) => {
                if output.tag.is_empty() {
                    return Err(DemBindingError::value(
                        "inferred output tag must be a non-empty string",
                    ));
                }
                let ids = supports
                    .get(&(output.tag.clone(), output.element))
                    .ok_or_else(|| {
                        DemBindingError::value(format!(
                            "{context} requires inferred output tag {:?} element {}",
                            output.tag, output.element
                        ))
                    })?;
                if ids.iter().any(|id| !index.runtime_ids.contains(id)) {
                    return Err(DemBindingError::value(format!(
                        "{context} contains unknown inferred measurement IDs"
                    )));
                }
                (ids.clone(), DemAnnotationOrigin::Inferred(output.clone()))
            }
        };
        let ids = xor_normalize(&ids, &context)?;
        let mut entry = spec.metadata.clone();
        validate_metadata(&entry)?;
        if let DemSpecSource::Inferred(output) = &spec.source {
            entry.insert("inferred_from_result_tag".into(), json!(output.tag));
            entry.insert("result_element".into(), json!(output.element));
        }
        entries.push(ResolvedDemSpec {
            id,
            measurement_ids: ids,
            coords: if role == "detector" {
                spec.coords.clone()
            } else {
                None
            },
            metadata: entry,
        });
        origins.push(DemDefinitionOrigin {
            role,
            id,
            source: origin,
        });
    }
    let ids = entries
        .iter()
        .map(|entry| entry.id)
        .collect::<BTreeSet<_>>();
    if role == "detector" {
        if ids.len() != entries.len() || ids.iter().copied().ne(0..entries.len()) {
            return Err(DemBindingError::value(
                "detector ids must be unique and dense from 0 to len(detectors)-1",
            ));
        }
    } else if ids.len() != entries.len() {
        return Err(DemBindingError::value("observable ids must be unique"));
    }
    entries.sort_by_key(|entry| entry.id);
    Ok(entries)
}

fn xor_normalize(ids: &[usize], context: &str) -> Result<Vec<usize>> {
    let mut odd = BTreeSet::new();
    for id in ids {
        if !odd.insert(*id) {
            odd.remove(id);
        }
    }
    let normalized: Vec<_> = ids.iter().copied().filter(|id| odd.remove(id)).collect();
    if normalized.is_empty() {
        return Err(DemBindingError::value(format!(
            "{context} references every measurement with even multiplicity; its parity is identically zero"
        )));
    }
    Ok(normalized)
}

pub(super) fn validate_metadata(metadata: &Map<String, Value>) -> Result<()> {
    let reserved: Vec<_> = [
        "coords",
        "id",
        "inferred_from_result_tag",
        "meas_ids",
        "records",
        "result_element",
        "result_tags",
    ]
    .into_iter()
    .filter(|key| metadata.contains_key(*key))
    .collect();
    if !reserved.is_empty() {
        return Err(DemBindingError::value(format!(
            "detector/observable metadata uses reserved keys: {reserved:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
