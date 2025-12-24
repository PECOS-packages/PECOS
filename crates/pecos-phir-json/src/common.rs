use pecos_core::errors::PecosError;

/// Versions of the PHIR-JSON format specification supported by this crate
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PhirJsonVersion {
    /// PHIR-JSON v0.1.0 (initial version)
    V0_1,
    // Add future versions here
}

/// Detects which version of PHIR-JSON is being used by examining the "version" field in the input JSON
///
/// # Errors
///
/// Returns an error if the JSON cannot be parsed or the version is unsupported.
pub fn detect_version(json: &str) -> Result<PhirJsonVersion, PecosError> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(|e| {
        PecosError::Input(format!(
            "Failed to parse PHIR-JSON program: Invalid JSON format: {e}"
        ))
    })?;

    if let Some(version) = value.get("version").and_then(|v| v.as_str()) {
        match version {
            "0.1.0" => Ok(PhirJsonVersion::V0_1),
            // Add future versions here
            _ => Err(PecosError::Input(format!(
                "Unsupported PHIR-JSON version: {version}"
            ))),
        }
    } else {
        Err(PecosError::Input(
            "Missing version field in PHIR-JSON program".into(),
        ))
    }
}
