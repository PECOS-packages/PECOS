// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Value-only named measurement bindings and certificate verification.

use super::specifications::{MeasurementRef, ResultRef};
#[cfg(feature = "selene")]
pub use pecos_qis::text::ResultTagMeasurements;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

/// Certification failures alone may be discarded during optional enrichment.
/// Python object-shape and integer conversion errors precede these value errors.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DemBindingError {
    #[error("{0}")]
    Value(String),
    #[error("{0}")]
    Certification(String),
    #[error(
        "Cannot bind result_ref: the program's lowered form could not be certified; use records, rec[-k], or meas_ids instead. {source}"
    )]
    RequiredCertification {
        #[source]
        source: Box<DemBindingError>,
    },
}

impl DemBindingError {
    pub(super) fn value(message: impl Into<String>) -> Self {
        Self::Value(message.into())
    }
}

type Result<T> = std::result::Result<T, DemBindingError>;
/// The compiler result-tag map and static measurement count, without requiring LLVM.
/// This is the value returned by `pecos_qis::text::extract_result_tag_measurements`.
#[cfg(not(feature = "selene"))]
pub type ResultTagMeasurements = (BTreeMap<String, Vec<Option<usize>>>, usize);

/// One runtime `result()` call after Python object conversion.
/// Non-string names are omitted at the boundary. Non-list values, mismatched
/// widths, non-list IDs and empty IDs map to `measurement_ids: None`
/// (unsupported). `width` is the actual list length, or zero for non-list values;
/// indexing and compiler holes use at least one slot. Present IDs have `width`
/// elements. Python rejects booleans, non-integers and negative occurrences or
/// measurement IDs before converting to `usize`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultTrace {
    pub name: String,
    pub occurrence: Option<usize>,
    pub width: usize,
    pub measurement_ids: Option<Vec<usize>>,
}

pub(super) fn numbered_occurrences(
    traces: &[ResultTrace],
) -> impl Iterator<Item = Result<usize>> + '_ {
    let mut next_occurrence = BTreeMap::<&str, Option<usize>>::new();
    traces.iter().map(move |trace| {
        let next = next_occurrence.entry(&trace.name).or_insert(Some(0));
        let occurrence = trace.occurrence.or(*next).ok_or_else(|| {
            DemBindingError::value(format!(
                "result trace {:?} has no available occurrence after usize::MAX",
                trace.name
            ))
        })?;
        *next = next.and_then(|next| occurrence.checked_add(1).map(|n| next.max(n)));
        Ok(occurrence)
    })
}

/// One runtime identity, its positions, and every result reference in encounter order.
/// Python checks runtime IDs are non-boolean, non-negative integers before conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasurementLedgerEntry {
    pub meas_id: usize,
    pub runtime_record_index: usize,
    pub canonical_record_index: Option<usize>,
    pub result_refs: Vec<ResultRef>,
}

impl MeasurementLedgerEntry {
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "meas_id": self.meas_id,
            "runtime_record_index": self.runtime_record_index,
            "canonical_record_index": self.canonical_record_index,
            "result_refs": self.result_refs.iter().map(ResultRef::to_json).collect::<Vec<_>>(),
        })
    }
}

/// Indexed result calls with holes and reverse measurement references.
/// Python validates runtime integer IDs and converts calls as documented on `ResultTrace`.
#[derive(Clone, Debug)]
pub struct ResultTraceIndex {
    calls: BTreeMap<String, BTreeMap<usize, Vec<Option<usize>>>>,
    refs_by_id: BTreeMap<usize, Vec<ResultRef>>,
    runtime_order: Vec<usize>,
    pub(super) runtime_ids: BTreeSet<usize>,
}

impl ResultTraceIndex {
    /// # Errors
    /// Rejects duplicate runtime identities before repeated or exhausted occurrences.
    pub fn new(runtime_order: &[usize], traces: &[ResultTrace]) -> Result<Self> {
        let mut runtime_ids = BTreeSet::new();
        let mut duplicates = BTreeSet::new();
        for id in runtime_order {
            if !runtime_ids.insert(*id) {
                duplicates.insert(*id);
            }
        }
        if !duplicates.is_empty() {
            return Err(DemBindingError::value(format!(
                "traced circuit contains duplicate MeasId(s): {:?}",
                duplicates.into_iter().take(8).collect::<Vec<_>>()
            )));
        }
        let mut index = Self {
            calls: BTreeMap::new(),
            refs_by_id: BTreeMap::new(),
            runtime_order: runtime_order.to_vec(),
            runtime_ids,
        };
        for (trace, occurrence) in traces.iter().zip(numbered_occurrences(traces)) {
            let occurrence = occurrence?;
            let calls = index.calls.entry(trace.name.clone()).or_default();
            if calls.contains_key(&occurrence) {
                return Err(DemBindingError::value(format!(
                    "result trace {:?} repeats occurrence {occurrence}",
                    trace.name
                )));
            }
            let ids = match &trace.measurement_ids {
                Some(ids) if !ids.is_empty() => {
                    for (element, id) in ids.iter().enumerate() {
                        index
                            .refs_by_id
                            .entry(*id)
                            .or_default()
                            .push(ResultRef::indexed(
                                trace.name.clone(),
                                occurrence,
                                (ids.len() > 1).then_some(element),
                            )?);
                    }
                    ids.iter().copied().map(Some).collect()
                }
                _ => vec![None; trace.width.max(1)],
            };
            calls.insert(occurrence, ids);
        }
        Ok(index)
    }

    fn dense(&self) -> bool {
        self.runtime_ids
            .iter()
            .copied()
            .eq(0..self.runtime_order.len())
    }

    /// # Errors
    /// Rejects noncanonical records, missing calls, holes, arrays and scalar elements.
    pub(crate) fn resolve_ref(&self, reference: &MeasurementRef) -> Result<usize> {
        match reference {
            MeasurementRef::Record(offset) => {
                if !self.dense() {
                    return Err(DemBindingError::value(
                        "rec[...] requires runtime MeasIds to preserve the canonical dense Guppy measurement identity range; use result_ref(...) for non-dense runtimes",
                    ));
                }
                let count = self.runtime_order.len();
                count.checked_sub(offset.get()).ok_or_else(|| {
                    DemBindingError::value(format!(
                        "measurement record offset -{offset} is out of range for {count} measurements"
                    ))
                })
            }
            MeasurementRef::Result(reference) => {
                let tag = reference.tag();
                let occurrence = reference.occurrence();
                let ids = self.calls.get(tag).and_then(|calls| calls.get(&occurrence)).ok_or_else(|| DemBindingError::value(format!("result_ref {tag:?} occurrence {occurrence} is absent from the runtime trace")))?;
                if ids.iter().any(Option::is_none) {
                    return Err(DemBindingError::value(format!(
                        "result_ref {tag:?} occurrence {occurrence} is not a direct scalar measurement result"
                    )));
                }
                let [Some(id)] = ids.as_slice() else {
                    return Err(DemBindingError::value(
                        "array-valued result_ref provenance is not yet certified; expose scalar result() tags or use canonical rec[...] references",
                    ));
                };
                if reference.element().is_some_and(|element| element != 0) {
                    return Err(DemBindingError::value(format!(
                        "scalar result_ref {tag:?} only accepts element=None or element=0"
                    )));
                }
                Ok(*id)
            }
        }
    }

    /// Sorted tags, dense call occurrences, flattened with unsupported holes.
    /// Only the independently bound raw tag may carry supported array calls.
    #[must_use]
    pub fn named_bindings(
        &self,
        raw_array_tag: Option<&str>,
    ) -> BTreeMap<String, Vec<Option<usize>>> {
        self.calls
            .iter()
            .filter(|(tag, calls)| {
                !calls.is_empty()
                    && calls.keys().copied().eq(0..calls.len())
                    && (Some(tag.as_str()) == raw_array_tag
                        || calls
                            .values()
                            .all(|call| call.len() == 1 || call.iter().all(Option::is_none)))
            })
            .map(|(tag, calls)| (tag.clone(), calls.values().flatten().copied().collect()))
            .collect()
    }

    #[must_use]
    pub fn ledger(&self) -> Vec<MeasurementLedgerEntry> {
        let dense = self.dense();
        self.runtime_order
            .iter()
            .enumerate()
            .map(|(position, id)| MeasurementLedgerEntry {
                meas_id: *id,
                runtime_record_index: position,
                canonical_record_index: dense.then_some(*id),
                result_refs: self.refs_by_id.get(id).cloned().unwrap_or_default(),
            })
            .collect()
    }
}

/// Validate the source order against the runtime set, preserving source order.
/// The adapter selects QIS metadata before Guppy metadata, defaulting to 0..N.
/// # Errors
/// Rejects duplicates in either order or unequal sets.
pub(crate) fn validate_source_ids(source: &[usize], runtime: &[usize]) -> Result<()> {
    let source_set = source.iter().copied().collect::<BTreeSet<_>>();
    let runtime_set = runtime.iter().copied().collect::<BTreeSet<_>>();
    if source_set.len() != source.len()
        || runtime_set.len() != runtime.len()
        || source_set != runtime_set
    {
        return Err(DemBindingError::value(
            "source and runtime measurement identities must be unique and describe the same set",
        ));
    }
    Ok(())
}

/// Certify scalar calls using lazily supplied compiler result-tag analysis. No candidate tags
/// means no analysis or identity validation. Only certification errors may be
/// dropped for optional enrichment; unrelated errors always propagate.
/// # Errors
/// Preserves identity/value errors and wraps required certification failures.
pub fn compiler_certified_traces(
    source: &[usize],
    runtime: &[usize],
    traces: &[ResultTrace],
    required_tags: &[String],
    analysis: impl FnOnce() -> Result<ResultTagMeasurements>,
) -> Result<Vec<ResultTrace>> {
    let candidates: BTreeSet<_> = required_tags
        .iter()
        .chain(traces.iter().map(|trace| &trace.name))
        .collect();
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let mut widths = BTreeMap::new();
    let mut next = BTreeMap::<&str, usize>::new();
    for trace in traces {
        let occurrence = next.entry(&trace.name).or_default();
        widths.insert((trace.name.as_str(), *occurrence), trace.width.max(1));
        *occurrence += 1;
    }
    let analysis = analysis().and_then(|(tags, count)| {
        validate_source_ids(source, runtime)?;
        if count != source.len() {
            return Err(DemBindingError::Certification(format!("Cannot certify measurement order from the lowered program: the certified lowered program has {count} measurement(s) but the traced program emitted {} measurement(s)", source.len())));
        }
        Ok(tags)
    });
    let tags = match analysis {
        Err(error @ DemBindingError::Certification(_)) => {
            if required_tags.is_empty() {
                return Ok(Vec::new());
            }
            return Err(DemBindingError::RequiredCertification {
                source: Box::new(error),
            });
        }
        other => other?,
    };
    let mut certified = Vec::new();
    for tag in candidates {
        let Some(occurrences) = tags.get(tag) else {
            if required_tags.contains(tag) {
                return Err(DemBindingError::value(format!(
                    "result_ref {tag:?} is absent from the compiled Guppy program"
                )));
            }
            continue;
        };
        for (occurrence, ordinal) in occurrences.iter().enumerate() {
            let (width, ids) = if let Some(ordinal) = ordinal {
                let id = source.get(*ordinal).ok_or_else(|| DemBindingError::value(format!("compiler measurement ordinal {ordinal} is outside the traced source measurement stream")))?;
                (1, Some(vec![*id]))
            } else {
                (*widths.get(&(tag.as_str(), occurrence)).unwrap_or(&1), None)
            };
            certified.push(ResultTrace {
                name: tag.clone(),
                occurrence: Some(occurrence),
                width,
                measurement_ids: ids,
            });
        }
    }
    Ok(certified)
}

/// Adapt the existing compiler result-tag extractor without changing its error category.
#[cfg(feature = "selene")]
impl From<pecos_qis::text::ResultTagCertificationError> for DemBindingError {
    fn from(error: pecos_qis::text::ResultTagCertificationError) -> Self {
        Self::Certification(error.to_string())
    }
}

/// Validate a generator's flattened slots, then transport source identities.
/// Python validates sequence/tuple/string shapes and non-boolean, non-negative
/// integer layout slots and measurement IDs before conversion.
/// # Errors
/// Rejects source identity mismatches, layout length, repeated slots, incorrect
/// widths, missing required tags and contradictory runtime provenance, in order.
pub fn generator_certified_traces(
    layout: &[(String, usize)],
    source: &[usize],
    runtime: &[usize],
    traces: &[ResultTrace],
    required_tags: &[String],
) -> Result<Vec<ResultTrace>> {
    validate_source_ids(source, runtime)?;
    if layout.len() != source.len() {
        return Err(DemBindingError::value(format!(
            "Guppy named measurement layout has {} entries but the source trace has {} measurements",
            layout.len(),
            source.len()
        )));
    }
    let mut runtime_counts = BTreeMap::<&str, usize>::new();
    for trace in traces {
        *runtime_counts.entry(&trace.name).or_default() += trace.width;
    }
    let mut counts = Vec::<(&str, u128)>::new();
    let mut seen = BTreeSet::new();
    for (tag, slot) in layout {
        if !seen.insert((tag, slot)) {
            return Err(DemBindingError::value(format!(
                "Guppy named measurement layout repeats output slot {:?}",
                (tag, slot)
            )));
        }
        if let Some((_, count)) = counts.iter_mut().find(|(name, _)| *name == tag) {
            *count = (*count).max((*slot as u128) + 1);
        } else {
            counts.push((tag, (*slot as u128) + 1));
        }
    }
    for (tag, count) in &counts {
        let actual = runtime_counts.get(tag).copied();
        if actual.map(|n| n as u128) != Some(*count) {
            return Err(DemBindingError::value(format!(
                "Guppy named measurement layout expects {count} value(s) for {tag:?}, but the runtime trace emits {}",
                actual.unwrap_or(0)
            )));
        }
    }
    let missing: BTreeSet<_> = required_tags
        .iter()
        .filter(|tag| !counts.iter().any(|(name, _)| name == tag))
        .collect();
    if !missing.is_empty() {
        return Err(DemBindingError::value(format!(
            "result_ref tag(s) are absent from the generator-certified layout: {:?}",
            missing.into_iter().collect::<Vec<_>>()
        )));
    }
    let mut runtime_slots = BTreeMap::new();
    let mut offsets = BTreeMap::<&str, usize>::new();
    for trace in traces {
        let offset = offsets.entry(&trace.name).or_default();
        let start = *offset;
        *offset += trace.width;
        if let Some(ids) = &trace.measurement_ids {
            for (element, id) in ids.iter().enumerate() {
                let slot = start + element;
                runtime_slots.insert((trace.name.as_str(), slot), *id);
            }
        }
    }
    for ((tag, slot), expected) in layout.iter().zip(source) {
        let mut actual = runtime_slots.get(&(tag.as_str(), *slot));
        if actual.is_none()
            && *slot == 0
            && let Some(synthetic) = synthetic_slot(tag)
        {
            actual = synthetic
                .element
                .and_then(|element| runtime_slots.get(&(synthetic.base, element)));
        }
        if let Some(actual) = actual
            && actual != expected
        {
            return Err(DemBindingError::value(format!(
                "runtime result trace {tag:?}[{slot}] has measurement id {actual}, but the generator-certified layout requires {expected}"
            )));
        }
    }
    Ok(layout
        .iter()
        .zip(source)
        .map(|((tag, slot), id)| ResultTrace {
            name: tag.clone(),
            occurrence: Some(*slot),
            width: 1,
            measurement_ids: Some(vec![*id]),
        })
        .collect())
}

/// Parsed generator tag; Python checks layout tags are strings before conversion.
#[derive(Debug, PartialEq, Eq)]
struct SyntheticSlot<'a> {
    base: &'a str,
    digits: &'a str,
    // Overflow is an out-of-range index, not an unrecognized tag.
    element: Option<usize>,
}

fn synthetic_slot(tag: &str) -> Option<SyntheticSlot<'_>> {
    let (base, digits) = tag.rsplit_once(":meas:")?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some(SyntheticSlot {
        base,
        digits,
        element: digits.parse().ok(),
    })
}

/// Cross-check independently certified raw slots against an inference binding.
/// Compiler occurrences address calls; generator occurrences address flat slots.
/// # Errors
/// Rejects call-width disagreements or mismatched IDs, including absent slots.
/// Rejects certified records without explicit occurrences.
pub fn check_certified_raw_binding(
    certified: &[ResultTrace],
    bound: &[ResultTrace],
    raw_tag: &str,
    generator_slots: bool,
) -> Result<()> {
    let calls: Vec<_> = bound
        .iter()
        .filter(|call| call.name == raw_tag)
        .map(|call| call.measurement_ids.as_deref().unwrap_or_default())
        .collect();
    let bound_ids: Vec<_> = calls.iter().flat_map(|call| call.iter().copied()).collect();
    let mut starts = Vec::new();
    let mut offset = 0;
    for call in &calls {
        starts.push(offset);
        offset += call.len();
    }
    for item in certified {
        let Some(ids) = &item.measurement_ids else {
            continue;
        };
        if ids.is_empty() {
            continue;
        }
        let occurrence = item.occurrence.ok_or_else(|| {
            DemBindingError::value(format!(
                "certified result trace {:?} has no explicit occurrence",
                item.name
            ))
        })?;
        let position = if !generator_slots && item.name == raw_tag {
            if let Some(call) = calls.get(occurrence)
                && call.len() != ids.len()
            {
                return Err(DemBindingError::value(format!(
                    "raw tag {raw_tag:?} call {occurrence} emits {} value(s), but its certificate names {} measurement(s)",
                    call.len(),
                    ids.len()
                )));
            }
            starts.get(occurrence).copied().unwrap_or(bound_ids.len())
        } else if generator_slots && item.name == raw_tag {
            occurrence
        } else if generator_slots
            && occurrence == 0
            && let Some(synthetic) = synthetic_slot(&item.name)
            && synthetic.base == raw_tag
        {
            let Some(element) = synthetic.element else {
                return Err(DemBindingError::value(format!(
                    "raw tag {raw_tag:?} element {:?}: certified measurement ID {} disagrees with inference binding ID None",
                    synthetic.digits, ids[0]
                )));
            };
            element
        } else {
            continue;
        };
        for (element, identity) in ids.iter().enumerate() {
            // An out-of-range position fails on the first element, before addition
            // could overflow. Successful positions are bounded by the vector length.
            let index = position + element;
            let inferred = bound_ids.get(index);
            if inferred != Some(identity) {
                return Err(DemBindingError::value(format!(
                    "raw tag {raw_tag:?} element {index}: certified measurement ID {identity} disagrees with inference binding ID {inferred:?}"
                )));
            }
        }
    }
    Ok(())
}

/// Inference evidence takes precedence over generator and compiler bindings.
#[must_use]
pub fn named_binding_classification<'a>(
    inference_binding: Option<&'a str>,
    generator_layout: bool,
    bindings: &BTreeMap<String, Vec<Option<usize>>>,
) -> &'a str {
    if let Some(binding) = inference_binding {
        return binding;
    }
    if generator_layout {
        return "generator_layout_v2_program_bound";
    }
    let mut present = false;
    let mut holes = false;
    for id in bindings.values().flatten() {
        if id.is_some() {
            present = true;
        } else {
            holes = true;
        }
    }
    match (present, holes) {
        (false, _) => "none",
        (true, true) => "compiler_direct_scalar_partial",
        (true, false) => "compiler_direct_scalar_complete",
    }
}

#[cfg(test)]
mod tests;
