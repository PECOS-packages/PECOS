//! External dependency definitions and extraction
//!
//! This module provides dependency information by reading from pecos.toml
//! and utilities for ensuring dependencies are downloaded and extracted.
//!
//! # Directory Structure
//!
//! ```text
//! ~/.pecos/
//! ├── cache/      # Downloaded archives (tar.gz, etc.)
//! ├── deps/       # Extracted source trees (ready for building)
//! └── ...
//! ```
//!
//! # Usage
//!
//! Build scripts should use `ensure_dep_ready()` to get a dependency:
//!
//! ```no_run
//! # use pecos_build::{Manifest, ensure_dep_ready};
//! # fn main() -> pecos_build::Result<()> {
//! let manifest = Manifest::find_and_load_validated()?;
//! let qulacs_path = ensure_dep_ready("qulacs", &manifest)?;
//! # Ok(())
//! # }
//! ```

use crate::download::download_cached;
use crate::errors::Result;
use crate::extract::extract_to_deps;
use crate::home::get_deps_dir;
use crate::manifest::Manifest;
use std::path::PathBuf;

/// Information about an available dependency
#[derive(Debug, Clone)]
pub struct DependencyInfo {
    /// Name of the dependency
    pub name: String,
    /// Version or commit
    pub version: String,
    /// Description
    pub description: String,
}

/// List all available dependencies from the manifest
#[must_use]
pub fn list_dependencies() -> Vec<DependencyInfo> {
    let manifest = Manifest::find_and_load().unwrap_or_else(|_| Manifest::default_pecos());

    manifest
        .dependencies
        .iter()
        .map(|(name, def)| {
            // Truncate commit hashes to 8 chars for display
            let version =
                if def.version.len() > 8 && def.version.chars().all(|c| c.is_ascii_hexdigit()) {
                    def.version[..8].to_string()
                } else {
                    def.version.clone()
                };

            DependencyInfo {
                name: name.clone(),
                version,
                description: def.description.clone().unwrap_or_default(),
            }
        })
        .collect()
}

/// Ensure a dependency is downloaded and extracted to `~/.pecos/deps/`
///
/// This is the primary function for build scripts to use. It will:
/// 1. Download the archive to `~/.pecos/cache/` if not already present
/// 2. Extract to `~/.pecos/deps/<name>-<version>/` if not already extracted
/// 3. Return the path to the extracted source tree
///
/// The extracted sources persist across `cargo clean`, so subsequent builds
/// don't need to re-download or re-extract.
///
/// # Arguments
///
/// * `name` - The dependency name (must be defined in the manifest)
/// * `manifest` - The loaded manifest containing dependency definitions
///
/// # Errors
///
/// Returns an error if:
/// - The dependency is not defined in the manifest
/// - Download fails
/// - Extraction fails
///
/// # Example
///
/// ```no_run
/// # use pecos_build::{Manifest, ensure_dep_ready};
/// # fn main() -> pecos_build::Result<()> {
/// let manifest = Manifest::find_and_load_validated()?;
/// let qulacs_path = ensure_dep_ready("qulacs", &manifest)?;
/// let eigen_path = ensure_dep_ready("eigen", &manifest)?;
/// # Ok(())
/// # }
/// ```
pub fn ensure_dep_ready(name: &str, manifest: &Manifest) -> Result<PathBuf> {
    // Get download info from manifest
    let info = manifest.get_download_info(name)?;

    // Check if already extracted
    let version_short = &info.version[..12.min(info.version.len())];
    let dep_dir_name = format!("{name}-{version_short}");
    let deps_dir = get_deps_dir()?;
    let dep_path = deps_dir.join(&dep_dir_name);

    if dep_path.exists() {
        // Already extracted, just return the path silently
        return Ok(dep_path);
    }

    // Download the archive (will be cached in ~/.pecos/cache/)
    log::info!("Downloading {name}...");
    let data = download_cached(&info)?;

    // Extract to deps directory
    log::info!("Extracting {name} to {}", dep_path.display());
    extract_to_deps(&data, &dep_dir_name)?;

    Ok(dep_path)
}
