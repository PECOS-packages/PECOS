//! cuQuantum configuration management for `.cargo/config.toml`

use crate::cuquantum::{
    find_cuquantum, get_lib_dir, get_pecos_cuquantum_dir, is_valid_cuquantum_installation,
};
use crate::errors::{Error, Result};
use crate::llvm::find_cargo_project_root;
use std::fs;
use std::path::{Path, PathBuf};

/// Result of validating the cuQuantum configuration
#[derive(Debug)]
pub struct ConfigValidation {
    /// Path configured in .cargo/config.toml (if any)
    pub configured_path: Option<PathBuf>,
    /// Whether the configured path exists
    pub path_exists: bool,
    /// Whether the configured path is valid cuQuantum
    pub path_is_valid: bool,
    /// Path that `find_cuquantum` would return
    pub detected_path: Option<PathBuf>,
    /// Whether config matches detected cuQuantum
    pub config_matches_detected: bool,
}

impl ConfigValidation {
    /// Check if the configuration is healthy
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.configured_path.is_some() && self.path_exists && self.path_is_valid
    }
}

/// Read the configured cuQuantum path from .cargo/config.toml
#[must_use]
#[allow(clippy::collapsible_if)]
pub fn read_configured_cuquantum_path() -> Option<PathBuf> {
    let project_root = find_cargo_project_root()?;
    let config_path = project_root.join(".cargo").join("config.toml");

    let content = fs::read_to_string(&config_path).ok()?;

    // Parse out CUQUANTUM_ROOT value
    // Handles both formats:
    //   CUQUANTUM_ROOT = "/path/to/cuquantum"
    //   CUQUANTUM_ROOT = { value = "/path/to/cuquantum", force = true }
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("CUQUANTUM_ROOT") {
            if let Some(eq_pos) = trimmed.find('=') {
                let value_part = trimmed[eq_pos + 1..].trim();

                // Check for inline table format: { value = "...", ... }
                if value_part.starts_with('{') {
                    if let Some(value_start) = value_part.find("value") {
                        let after_value = &value_part[value_start + 5..];
                        if let Some(eq_pos) = after_value.find('=') {
                            let path_part = after_value[eq_pos + 1..].trim();
                            // Extract quoted string
                            if let Some(start) = path_part.find('"') {
                                if let Some(end) = path_part[start + 1..].find('"') {
                                    let path = &path_part[start + 1..start + 1 + end];
                                    return Some(PathBuf::from(path));
                                }
                            }
                        }
                    }
                } else {
                    // Simple format: "..."
                    if let Some(start) = value_part.find('"') {
                        if let Some(end) = value_part[start + 1..].find('"') {
                            let path = &value_part[start + 1..start + 1 + end];
                            return Some(PathBuf::from(path));
                        }
                    }
                }
            }
        }
    }

    None
}

/// Validate the current cuQuantum configuration
#[must_use]
pub fn validate_cuquantum_config() -> ConfigValidation {
    let configured_path = read_configured_cuquantum_path();
    let detected_path = find_cuquantum();

    let (path_exists, path_is_valid) = if let Some(ref path) = configured_path {
        (path.exists(), is_valid_cuquantum_installation(path))
    } else {
        (false, false)
    };

    let config_matches_detected = match (&configured_path, &detected_path) {
        (Some(configured), Some(detected)) => configured == detected,
        (None, None) => true,
        _ => false,
    };

    ConfigValidation {
        configured_path,
        path_exists,
        path_is_valid,
        detected_path,
        config_matches_detected,
    }
}

/// Automatically configure cuQuantum for PECOS
///
/// This function determines the best cuQuantum installation to use and writes
/// it to `.cargo/config.toml` with `force=true`.
///
/// Priority order:
/// 1. `~/.pecos/cuquantum` (PECOS-managed cuQuantum)
/// 2. `CUQUANTUM_ROOT` environment variable
/// 3. System cuQuantum (standard paths, etc.)
///
/// # Errors
///
/// Returns an error if no suitable cuQuantum installation could be found
pub fn auto_configure_cuquantum(project_root: Option<PathBuf>) -> Result<PathBuf> {
    // Priority 1: Check ~/.pecos/deps/cuquantum for PECOS-managed cuQuantum
    if let Ok(pecos_cuquantum) = get_pecos_cuquantum_dir()
        && is_valid_cuquantum_installation(&pecos_cuquantum)
    {
        let project_root = project_root
            .or_else(find_cargo_project_root)
            .ok_or_else(|| Error::Config("Could not find Cargo project root".into()))?;

        write_cargo_config(&project_root, &pecos_cuquantum, true)?;
        return Ok(pecos_cuquantum);
    }

    // Priority 2: Check CUQUANTUM_ROOT env var
    if let Ok(cuquantum_root) = std::env::var("CUQUANTUM_ROOT") {
        let path = PathBuf::from(&cuquantum_root);
        if is_valid_cuquantum_installation(&path) {
            let project_root = project_root
                .or_else(find_cargo_project_root)
                .ok_or_else(|| Error::Config("Could not find Cargo project root".into()))?;

            write_cargo_config(&project_root, &path, true)?;
            return Ok(path);
        }
    }

    // Priority 3: Scan system for cuQuantum
    if let Some(detected_path) = find_cuquantum() {
        let project_root = project_root
            .or_else(find_cargo_project_root)
            .ok_or_else(|| Error::Config("Could not find Cargo project root".into()))?;

        write_cargo_config(&project_root, &detected_path, true)?;
        return Ok(detected_path);
    }

    Err(Error::CuQuantum(
        "No suitable cuQuantum installation found".into(),
    ))
}

/// Write or update `.cargo/config.toml` with cuQuantum configuration
///
/// This sets:
/// - `CUQUANTUM_ROOT` - Path to cuQuantum installation (for build scripts)
///
/// # Arguments
/// * `project_root` - Path to the Cargo project root
/// * `cuquantum_path` - Path to the cuQuantum installation
/// * `force` - If true, use `force=true` to override shell environment variables
///
/// # Errors
///
/// Returns an error if the `.cargo` directory cannot be created or the config file
/// cannot be written.
pub fn write_cargo_config(project_root: &Path, cuquantum_path: &Path, force: bool) -> Result<()> {
    // Forward slashes keep the value backslash-escape-free in TOML.
    let cuquantum_path_str = cuquantum_path.to_string_lossy().replace('\\', "/");
    let mut cfg = crate::cargo_config::CargoConfig::open(project_root)?;
    cfg.set_env("CUQUANTUM_ROOT", &cuquantum_path_str, force)?;
    cfg.save()?;
    Ok(())
}

/// Get the library path string for cuQuantum (for `LD_LIBRARY_PATH` hints)
#[must_use]
pub fn get_library_path_hint(cuquantum_path: &Path) -> Option<String> {
    let lib_dir = get_lib_dir(cuquantum_path)?;
    Some(lib_dir.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_returns_struct() {
        // This test just verifies the function runs without panic
        let _validation = validate_cuquantum_config();
    }
}
