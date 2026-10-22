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
                .ok_or_else(|| error("expected an object or null"))?;
            let block = object
                .get("block")
                .and_then(serde_json::Value::as_str)
                .filter(|block| !block.is_empty())
                .ok_or_else(|| error("block must be a non-empty string"))?;
            let family = match object.get("family") {
                Some(serde_json::Value::Null) => None,
                Some(serde_json::Value::String(family)) if !family.is_empty() => {
                    Some(family.clone())
                }
                _ => {
                    return Err(error(
                        "family is required and must be a non-empty string or null",
                    ));
                }
            };
            let round = object
                .get("round")
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| error("round must be a signed 64-bit integer"))?;
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
