//! PECOS dependency manifest support
//!
//! This module provides `pecos.toml` manifest support for tracking PECOS dependencies.
//! Each crate that needs external C++ libraries ships its own `pecos.toml` as the
//! source of truth. Use `pecos deps check` to verify consistency across crates.
//!
//! # Why No Lock File?
//!
//! Unlike Cargo (which needs a lock file to resolve version ranges), our manifest
//! already specifies exact URLs and SHA256 checksums. There's no resolution step,
//! so `pecos.toml` effectively serves as both manifest AND lock file.
//!
//! # Structure
//!
//! ```toml
//! version = 1
//!
//! [dependencies.quest]
//! version = "v4.2.0"
//! url = "https://github.com/QuEST-Kit/QuEST/archive/refs/tags/v4.2.0.tar.gz"
//! sha256 = "2c812a7ec4d727e0947ffd0daf05452963c3f1c10e428c8bc30c35164921fcba"
//! ```
//!
//! # File Location
//!
//! Per-crate: `crates/<name>/pecos.toml` - shipped with each crate

use crate::errors::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

// =============================================================================
// Manifest (pecos.toml)
// =============================================================================

/// PECOS dependency manifest (`pecos.toml`)
///
/// Specifies which crates need which dependencies and defines all available dependencies.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    /// Manifest format version
    #[serde(default = "default_manifest_version")]
    pub version: u32,

    /// LLVM configuration
    #[serde(default)]
    pub llvm: LlvmConfig,

    /// Dependency definitions (versions, URLs, checksums)
    #[serde(default)]
    pub dependencies: BTreeMap<String, DependencyDef>,
}

fn default_manifest_version() -> u32 {
    1
}

/// LLVM configuration in the manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlvmConfig {
    /// Required LLVM version (e.g., "14")
    #[serde(default = "default_llvm_version")]
    pub version: String,

    /// Whether LLVM is required for the project
    #[serde(default = "default_true")]
    pub required: bool,

    /// Which crates require LLVM
    #[serde(default)]
    pub required_by: Vec<String>,
}

impl Default for LlvmConfig {
    fn default() -> Self {
        Self {
            version: default_llvm_version(),
            required: true,
            required_by: vec![],
        }
    }
}

fn default_llvm_version() -> String {
    "14".to_string()
}

fn default_true() -> bool {
    true
}

/// Dependency definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyDef {
    /// Version or commit hash
    pub version: String,

    /// Download URL (optional - can be derived from version for known deps)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// SHA256 checksum
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,

    /// Description of this dependency
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl Manifest {
    /// Load manifest from a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed as TOML.
    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("Failed to read manifest: {e}")))?;
        toml::from_str(&content)
            .map_err(|e| Error::Config(format!("Failed to parse manifest: {e}")))
    }

    /// Save manifest to a file
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest cannot be serialized or the file cannot be written.
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("Failed to serialize manifest: {e}")))?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Find manifest file
    ///
    /// Search order:
    /// 1. `CARGO_MANIFEST_DIR` (crate-local pecos.toml)
    /// 2. Current directory and parents
    #[must_use]
    pub fn find() -> Option<PathBuf> {
        // First check CARGO_MANIFEST_DIR (set during cargo build)
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let crate_manifest = PathBuf::from(&manifest_dir).join("pecos.toml");
            if crate_manifest.exists() {
                return Some(crate_manifest);
            }
        }

        // Fall back to searching current directory and parents
        let current_dir = std::env::current_dir().ok()?;
        let mut path = current_dir.as_path();

        loop {
            let manifest_path = path.join("pecos.toml");
            if manifest_path.exists() {
                return Some(manifest_path);
            }
            path = path.parent()?;
        }
    }

    /// Find and load manifest from the current directory or parents
    ///
    /// This is the primary entry point for build scripts.
    ///
    /// # Errors
    ///
    /// Returns an error if no `pecos.toml` is found, or if it cannot be parsed.
    pub fn find_and_load() -> Result<Self> {
        let path = Self::find().ok_or_else(|| {
            Error::Config("pecos.toml not found in current directory or parents".into())
        })?;
        Self::load(&path)
    }

    /// Find and load manifest, or use defaults if not found
    ///
    /// This is the recommended entry point for build scripts in published crates.
    /// It allows developers working in the PECOS repo to use their local `pecos.toml`,
    /// while users who install crates from crates.io get sensible defaults.
    #[must_use]
    pub fn find_or_default() -> Self {
        Self::find()
            .and_then(|path| Self::load(&path).ok())
            .unwrap_or_default()
    }

    /// Get download info for a dependency by name
    ///
    /// Returns a `DownloadInfo` struct suitable for use with `download_cached`.
    ///
    /// # Errors
    ///
    /// Returns an error if the dependency is not found or is missing a URL or SHA256.
    pub fn get_download_info(&self, name: &str) -> Result<crate::DownloadInfo> {
        let dep = self
            .dependencies
            .get(name)
            .ok_or_else(|| Error::Config(format!("Dependency '{name}' not found in pecos.toml")))?;

        let url = dep
            .url
            .clone()
            .ok_or_else(|| Error::Config(format!("Dependency '{name}' has no URL defined")))?;

        let sha256 = dep
            .sha256
            .clone()
            .ok_or_else(|| Error::Config(format!("Dependency '{name}' has no sha256 defined")))?;

        Ok(crate::DownloadInfo {
            name: name.to_string(),
            version: dep.version.clone(),
            url,
            sha256,
        })
    }

    /// Get download info for multiple dependencies
    ///
    /// # Errors
    ///
    /// Returns an error if any dependency is not found or is missing a URL or SHA256.
    pub fn get_download_infos(&self, names: &[&str]) -> Result<Vec<crate::DownloadInfo>> {
        names
            .iter()
            .map(|name| self.get_download_info(name))
            .collect()
    }

    /// Find and load manifest
    ///
    /// This is the primary entry point for build scripts. It finds the crate-level
    /// `pecos.toml` (via `CARGO_MANIFEST_DIR`) or walks up from the current directory.
    ///
    /// # Errors
    ///
    /// Returns an error if no manifest is found or if parsing fails.
    pub fn find_and_load_validated() -> Result<Self> {
        Self::find_and_load()
    }
}

// =============================================================================
// Cross-crate consistency checking
// =============================================================================

/// A mismatch found when checking consistency across per-crate manifests
#[derive(Debug)]
pub struct DepMismatch {
    pub dep_name: String,
    pub field: String,
    pub values: BTreeMap<String, String>,
}

/// Check that shared dependencies are consistent across all per-crate `pecos.toml` files
///
/// Walks `crates/*/pecos.toml` under `workspace_root`, collects all dependency
/// definitions, and reports any dependency that appears in multiple crates with
/// different version/url/sha256 values.
///
/// # Errors
///
/// Returns an error if any manifest file cannot be parsed.
pub fn check_consistency(workspace_root: &Path) -> Result<Vec<DepMismatch>> {
    let crates_dir = workspace_root.join("crates");
    let mut all_deps: BTreeMap<String, BTreeMap<String, DependencyDef>> = BTreeMap::new();

    let entries = fs::read_dir(&crates_dir)
        .map_err(|e| Error::Config(format!("Cannot read crates directory: {e}")))?;

    for entry in entries {
        let entry = entry?;
        let manifest_path = entry.path().join("pecos.toml");
        if !manifest_path.exists() {
            continue;
        }
        let crate_name = entry.file_name().to_string_lossy().to_string();
        let manifest = Manifest::load(&manifest_path)?;
        for (dep_name, dep_def) in manifest.dependencies {
            all_deps
                .entry(dep_name)
                .or_default()
                .insert(crate_name.clone(), dep_def);
        }
    }

    let mut mismatches = Vec::new();

    for (dep_name, crate_defs) in &all_deps {
        if crate_defs.len() < 2 {
            continue;
        }

        // Check version consistency
        let versions: BTreeMap<String, String> = crate_defs
            .iter()
            .map(|(c, d)| (c.clone(), d.version.clone()))
            .collect();
        if versions
            .values()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1
        {
            mismatches.push(DepMismatch {
                dep_name: dep_name.clone(),
                field: "version".to_string(),
                values: versions,
            });
        }

        // Check URL consistency
        let urls: BTreeMap<String, String> = crate_defs
            .iter()
            .map(|(c, d)| (c.clone(), d.url.clone().unwrap_or_default()))
            .collect();
        if urls
            .values()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1
        {
            mismatches.push(DepMismatch {
                dep_name: dep_name.clone(),
                field: "url".to_string(),
                values: urls,
            });
        }

        // Check SHA256 consistency
        let shas: BTreeMap<String, String> = crate_defs
            .iter()
            .map(|(c, d)| (c.clone(), d.sha256.clone().unwrap_or_default()))
            .collect();
        if shas
            .values()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1
        {
            mismatches.push(DepMismatch {
                dep_name: dep_name.clone(),
                field: "sha256".to_string(),
                values: shas,
            });
        }
    }

    Ok(mismatches)
}

/// Collect all dependencies across all per-crate `pecos.toml` files
///
/// Returns a merged map of dependency name to definition. If a dependency
/// appears in multiple crates, the first one found (alphabetically by crate
/// name) wins.
///
/// # Errors
///
/// Returns an error if any manifest file cannot be parsed.
pub fn collect_all_deps(workspace_root: &Path) -> Result<BTreeMap<String, DependencyDef>> {
    let crates_dir = workspace_root.join("crates");
    let mut all_deps: BTreeMap<String, DependencyDef> = BTreeMap::new();

    let entries = fs::read_dir(&crates_dir)
        .map_err(|e| Error::Config(format!("Cannot read crates directory: {e}")))?;

    let mut dirs: Vec<_> = entries.filter_map(std::result::Result::ok).collect();
    dirs.sort_by_key(std::fs::DirEntry::file_name);

    for entry in dirs {
        let manifest_path = entry.path().join("pecos.toml");
        if !manifest_path.exists() {
            continue;
        }
        let manifest = Manifest::load(&manifest_path)?;
        for (dep_name, dep_def) in manifest.dependencies {
            all_deps.entry(dep_name).or_insert(dep_def);
        }
    }

    Ok(all_deps)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_manifest_round_trip() {
        let mut manifest = Manifest::default();
        manifest.dependencies.insert(
            "quest".to_string(),
            DependencyDef {
                version: "v4.2.0".to_string(),
                url: Some("https://example.com/quest.tar.gz".to_string()),
                sha256: Some("abc123".to_string()),
                description: Some("test dep".to_string()),
            },
        );
        let toml_str = toml::to_string_pretty(&manifest).unwrap();
        let parsed: Manifest = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.version, manifest.version);
        assert_eq!(parsed.dependencies.len(), manifest.dependencies.len());
    }

    #[test]
    fn test_check_consistency_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let crates_dir = tmp.path().join("crates");

        // Two crates sharing "stim" with same version
        let crate_a = crates_dir.join("crate-a");
        let crate_b = crates_dir.join("crate-b");
        fs::create_dir_all(&crate_a).unwrap();
        fs::create_dir_all(&crate_b).unwrap();

        let toml_content = r#"
version = 1
[dependencies.stim]
version = "abc123"
url = "https://example.com/stim.tar.gz"
sha256 = "deadbeef"
"#;
        let mut f = fs::File::create(crate_a.join("pecos.toml")).unwrap();
        f.write_all(toml_content.as_bytes()).unwrap();
        let mut f = fs::File::create(crate_b.join("pecos.toml")).unwrap();
        f.write_all(toml_content.as_bytes()).unwrap();

        let mismatches = check_consistency(tmp.path()).unwrap();
        assert!(mismatches.is_empty());
    }

    #[test]
    fn test_check_consistency_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let crates_dir = tmp.path().join("crates");

        let crate_a = crates_dir.join("crate-a");
        let crate_b = crates_dir.join("crate-b");
        fs::create_dir_all(&crate_a).unwrap();
        fs::create_dir_all(&crate_b).unwrap();

        let mut f = fs::File::create(crate_a.join("pecos.toml")).unwrap();
        f.write_all(
            br#"
version = 1
[dependencies.stim]
version = "v1"
url = "https://example.com/stim-v1.tar.gz"
sha256 = "aaa"
"#,
        )
        .unwrap();

        let mut f = fs::File::create(crate_b.join("pecos.toml")).unwrap();
        f.write_all(
            br#"
version = 1
[dependencies.stim]
version = "v2"
url = "https://example.com/stim-v2.tar.gz"
sha256 = "bbb"
"#,
        )
        .unwrap();

        let mismatches = check_consistency(tmp.path()).unwrap();
        assert!(!mismatches.is_empty());
        assert!(
            mismatches
                .iter()
                .any(|m| m.dep_name == "stim" && m.field == "version")
        );
    }
}
