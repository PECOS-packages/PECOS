use pecos_core::errors::PecosError;

/// Versions of the PHIR specification supported by this crate
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PHIRVersion {
    /// PHIR v0.1 (initial version)
    V0_1,
    // Add future versions here
}

/// Detects which version of PHIR is being used by examining the "version" field in the input JSON
pub fn detect_version(json: &str) -> Result<PHIRVersion, PecosError> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(|e| {
        PecosError::Input(format!(
            "Failed to parse PHIR program: Invalid JSON format: {e}"
        ))
    })?;

    if let Some(version) = value.get("version").and_then(|v| v.as_str()) {
        match version {
            "0.1.0" => Ok(PHIRVersion::V0_1),
            // Add future versions here
            _ => Err(PecosError::Input(format!(
                "Unsupported PHIR version: {version}"
            ))),
        }
    } else {
        Err(PecosError::Input(
            "Missing version field in PHIR program".into(),
        ))
    }
}
