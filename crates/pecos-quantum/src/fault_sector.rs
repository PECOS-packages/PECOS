// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Optional fault-sector metadata shared by circuit writers and readers.

/// Read an opaque detector fault-sector label; absent and null mean unknown.
///
/// Labels are compared only for equality. Detectors with different labels belong
/// to different fault sectors: the generator asserts that faults it treats as
/// single-sector do not flip detectors in two sectors. A label is not the physical
/// Pauli of the measured stabilizer: for a Clifford-deformed code it names the
/// deformed (rotated) family. CSS generators use "X" and "Z".
///
/// # Errors
///
/// Rejects empty strings and non-string values, naming the detector.
pub fn parse_fault_sector(
    value: Option<&serde_json::Value>,
    detector_id: impl std::fmt::Display,
) -> Result<Option<String>, String> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(value)) if !value.is_empty() => Ok(Some(value.clone())),
        Some(value) => Err(format!(
            "detector {detector_id} fault_sector must be a non-empty string or null, got {value}"
        )),
    }
}

/// A classified record in absolute measurement-record order.
///
/// The base family belongs to the stabilizer register before logical frame
/// changes; a transversal H does not change it. `None` is classified but untyped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasurementClass {
    /// Opaque non-empty code-block name.
    pub block: String,
    /// Base check family, or a classified but untyped record.
    pub family: Option<String>,
    /// Global integer epoch, including negative initialization rounds.
    pub round: i64,
}

/// Parse measurement classes, ignoring unknown object keys like detector metadata.
///
/// # Errors
/// Rejects malformed JSON or entries, naming the offending record ordinal.
pub fn parse_measurement_classes(json: &str) -> Result<Vec<Option<MeasurementClass>>, String> {
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|error| format!("measurement_classes: malformed JSON: {error}"))?;
    let entries = value
        .as_array()
        .ok_or_else(|| "measurement_classes: expected a JSON list".to_string())?;
    entries
        .iter()
        .enumerate()
        .map(|(ordinal, value)| {
            let error = |message: &str| format!("measurement_classes ordinal {ordinal}: {message}");
            if value.is_null() {
                return Ok(None);
            }
            let object = value
                .as_object()
                .ok_or_else(|| error(&format!("expected an object or null, got {value}")))?;
            let required = |key: &str| {
                object
                    .get(key)
                    .ok_or_else(|| error(&format!("{key} is missing, got {value}")))
            };
            let block_value = required("block")?;
            let block = block_value
                .as_str()
                .filter(|block| !block.is_empty())
                .ok_or_else(|| {
                    error(&format!(
                        "block must be a non-empty string, got {block_value}"
                    ))
                })?;
            let family = match required("family")? {
                serde_json::Value::Null => None,
                serde_json::Value::String(family) if !family.is_empty() => Some(family.clone()),
                value => {
                    return Err(error(&format!(
                        "family must be a non-empty string or null, got {value}",
                    )));
                }
            };
            let round_value = required("round")?;
            let round = round_value.as_i64().ok_or_else(|| {
                error(&format!(
                    "round must be a signed 64-bit integer, got {round_value}"
                ))
            })?;
            Ok(Some(MeasurementClass {
                block: block.to_owned(),
                family,
                round,
            }))
        })
        .collect()
}

/// Check classes against the final number of emitted records.
///
/// # Errors
/// Rejects a list whose length differs from `num_measurements`.
pub fn validate_measurement_classes_length(
    classes: &[Option<MeasurementClass>],
    num_measurements: usize,
) -> Result<(), String> {
    if classes.len() != num_measurements {
        return Err(format!(
            "measurement_classes length {} does not match measurement count {num_measurements}",
            classes.len(),
        ));
    }
    Ok(())
}

/// Resolve a detector's effective fault sector from absolute record positions.
///
/// Even multiplicities cancel. The remaining records must all have the same
/// non-null family to derive a label; block and round do not affect it.
///
/// # Errors
/// Rejects out-of-range positions and explicit labels that differ from the
/// derived result (including no common family), naming the detector.
pub fn resolve_detector_fault_sector(
    detector_id: u32,
    explicit: Option<&str>,
    positions: &[usize],
    classes: &[Option<MeasurementClass>],
) -> Result<Option<String>, String> {
    let mut support = std::collections::BTreeSet::new();
    for &ordinal in positions {
        if ordinal >= classes.len() {
            return Err(format!(
                "detector D{detector_id} references measurement {ordinal}, outside the {} measurement classes",
                classes.len(),
            ));
        }
        if !support.insert(ordinal) {
            support.remove(&ordinal);
        }
    }
    let family = |ordinal: usize| {
        classes[ordinal]
            .as_ref()
            .and_then(|class| class.family.as_deref())
    };
    let derived = support
        .first()
        .and_then(|&ordinal| family(ordinal))
        .filter(|expected| {
            support
                .iter()
                .all(|&ordinal| family(ordinal) == Some(*expected))
        });
    if let Some(explicit) = explicit
        && derived != Some(explicit)
    {
        let derived = derived.map_or_else(
            || "no common family".to_owned(),
            |family| format!("{family:?}"),
        );
        return Err(format!(
            "detector D{detector_id} declares fault_sector {explicit:?} but its measurement classes give {derived}",
        ));
    }
    Ok(derived.map(str::to_owned))
}
