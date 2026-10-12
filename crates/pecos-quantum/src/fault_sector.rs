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
