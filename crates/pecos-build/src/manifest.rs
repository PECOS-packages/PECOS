//! PECOS dependency manifest support
//!
//! This module provides `pecos.toml` manifest support for tracking PECOS dependencies.
//!
//! # Why No Lock File?
//!
//! Unlike Cargo (which needs a lock file to resolve version ranges), our manifest
//! already specifies exact URLs and SHA256 checksums. There's no resolution step,
//! so `pecos.toml` effectively serves as both manifest AND lock file.
//!
//! # Workspace Validation
//!
//! When building in a workspace context, the crate-level `pecos.toml` is validated
//! against the workspace-level `pecos.toml` to ensure they stay in sync. If they
//! differ, the build fails with a helpful error message suggesting to run
//! `pecos deps sync`.
//!
//! # Structure
//!
//! The manifest uses a workspace-level approach with per-crate declarations:
//!
//! ```toml
//! version = 1
//!
//! [llvm]
//! version = "14"
//! required = true
//!
//! # Per-crate dependency declarations
//! [crates.pecos-quest]
//! dependencies = ["quest"]
//!
//! [crates.pecos-qulacs]
//! dependencies = ["qulacs", "eigen"]
//!
//! # Dependency definitions with exact URLs and checksums
//! [dependencies.quest]
//! version = "v4.1.0"
//! url = "https://github.com/QuEST-Kit/QuEST/archive/refs/tags/v4.1.0.tar.gz"
//! sha256 = "85aa95bba6457c4f4e93221f4c417d988588891a1f7cb211c307dfe81a10cadd"
//! ```
//!
//! # File Locations
//!
//! - Workspace: `PECOS/pecos.toml` - Master manifest for developers
//! - Per-crate: `crates/<name>/pecos.toml` - Published with crate for crates.io users

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

    /// Per-crate dependency declarations
    #[serde(default)]
    pub crates: BTreeMap<String, CrateConfig>,

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

/// Per-crate configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CrateConfig {
    /// List of dependency names this crate requires
    #[serde(default)]
    pub dependencies: Vec<String>,

    /// Whether this crate requires LLVM
    #[serde(default)]
    pub requires_llvm: bool,
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
    /// 1. `CARGO_MANIFEST_DIR` (crate-local pecos.toml for published crates)
    /// 2. Current directory and parents (workspace pecos.toml for developers)
    #[must_use]
    pub fn find() -> Option<PathBuf> {
        // First check CARGO_MANIFEST_DIR (set during cargo build)
        // This allows published crates to include their own pecos.toml
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let crate_manifest = PathBuf::from(&manifest_dir).join("pecos.toml");
            if crate_manifest.exists() {
                return Some(crate_manifest);
            }
        }

        // Fall back to searching current directory and parents
        // This finds workspace-level pecos.toml for developers
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
            .unwrap_or_else(Self::default_pecos)
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

    /// Get dependencies for a specific crate
    #[must_use]
    pub fn get_crate_dependencies(&self, crate_name: &str) -> Vec<&DependencyDef> {
        self.crates
            .get(crate_name)
            .map(|config| {
                config
                    .dependencies
                    .iter()
                    .filter_map(|name| self.dependencies.get(name))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if a crate requires LLVM
    #[must_use]
    pub fn crate_requires_llvm(&self, crate_name: &str) -> bool {
        self.crates.get(crate_name).is_some_and(|c| c.requires_llvm)
            || self.llvm.required_by.contains(&crate_name.to_string())
    }

    /// Get all crates that use a specific dependency
    #[must_use]
    pub fn get_dependency_users(&self, dep_name: &str) -> Vec<&str> {
        self.crates
            .iter()
            .filter(|(_, config)| config.dependencies.contains(&dep_name.to_string()))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Find the crate-level manifest (in `CARGO_MANIFEST_DIR`)
    #[must_use]
    pub fn find_crate_manifest() -> Option<PathBuf> {
        std::env::var("CARGO_MANIFEST_DIR")
            .ok()
            .map(|dir| PathBuf::from(dir).join("pecos.toml"))
            .filter(|p| p.exists())
    }

    /// Find the workspace-level manifest by walking up from a starting directory
    #[must_use]
    pub fn find_workspace_manifest(start_dir: &Path) -> Option<PathBuf> {
        let mut path = start_dir;

        // Walk up looking for pecos.toml with [crates.*] sections (workspace indicator)
        loop {
            let manifest_path = path.join("pecos.toml");
            if manifest_path.exists() {
                // Check if this looks like a workspace manifest (has [crates.*] section)
                if let Ok(content) = fs::read_to_string(&manifest_path)
                    && content.contains("[crates.")
                {
                    return Some(manifest_path);
                }
            }
            path = path.parent()?;
        }
    }

    /// Validate that a crate manifest matches the workspace manifest
    ///
    /// Returns `Ok(())` if they match or if there's no workspace manifest.
    ///
    /// # Errors
    ///
    /// Returns an error if the crate manifest differs from the workspace manifest,
    /// with a detailed message listing all mismatches.
    pub fn validate_against_workspace(
        crate_manifest: &Self,
        crate_manifest_path: &Path,
    ) -> Result<()> {
        // Try to find workspace manifest
        let crate_dir = crate_manifest_path.parent().unwrap_or(Path::new("."));
        let Some(workspace_path) = Self::find_workspace_manifest(crate_dir) else {
            return Ok(()); // No workspace, nothing to validate
        };

        // Don't validate if crate manifest IS the workspace manifest
        if crate_manifest_path == workspace_path {
            return Ok(());
        }

        let workspace = Self::load(&workspace_path)?;
        let mut mismatches = Vec::new();

        // Check each dependency in the crate manifest against workspace
        for (dep_name, crate_dep) in &crate_manifest.dependencies {
            if let Some(workspace_dep) = workspace.dependencies.get(dep_name) {
                // Compare version
                if crate_dep.version != workspace_dep.version {
                    mismatches.push(format!(
                        "  {dep_name}: version mismatch\n    crate:     {}\n    workspace: {}",
                        crate_dep.version, workspace_dep.version
                    ));
                }
                // Compare URL
                if crate_dep.url != workspace_dep.url {
                    mismatches.push(format!(
                        "  {dep_name}: URL mismatch\n    crate:     {}\n    workspace: {}",
                        crate_dep.url.as_deref().unwrap_or("(none)"),
                        workspace_dep.url.as_deref().unwrap_or("(none)")
                    ));
                }
                // Compare SHA256
                if crate_dep.sha256 != workspace_dep.sha256 {
                    mismatches.push(format!(
                        "  {dep_name}: SHA256 mismatch\n    crate:     {}\n    workspace: {}",
                        crate_dep.sha256.as_deref().unwrap_or("(none)"),
                        workspace_dep.sha256.as_deref().unwrap_or("(none)")
                    ));
                }
            } else {
                mismatches.push(format!(
                    "  {dep_name}: exists in crate manifest but not in workspace"
                ));
            }
        }

        if mismatches.is_empty() {
            Ok(())
        } else {
            Err(Error::Config(format!(
                "Crate manifest does not match workspace manifest!\n\n\
                 Crate manifest: {}\n\
                 Workspace manifest: {}\n\n\
                 Mismatches:\n{}\n\n\
                 Run 'cargo run -p pecos -- deps sync' to update crate manifests from workspace.",
                crate_manifest_path.display(),
                workspace_path.display(),
                mismatches.join("\n")
            )))
        }
    }

    /// Find and load manifest, validating against workspace if applicable
    ///
    /// This is the primary entry point for build scripts. It:
    /// 1. Finds the crate-level manifest (`CARGO_MANIFEST_DIR`) or workspace manifest
    /// 2. If a crate-level manifest exists and we're in a workspace, validates consistency
    /// 3. Returns the loaded manifest
    ///
    /// # Errors
    ///
    /// Returns an error if no manifest is found, if parsing fails, or if the crate
    /// manifest does not match the workspace manifest.
    pub fn find_and_load_validated() -> Result<Self> {
        let crate_manifest_path = Self::find_crate_manifest();

        if let Some(crate_path) = crate_manifest_path {
            let manifest = Self::load(&crate_path)?;
            Self::validate_against_workspace(&manifest, &crate_path)?;
            Ok(manifest)
        } else {
            // No crate manifest, try to find workspace or any manifest
            Self::find_and_load()
        }
    }

    /// Generate a crate-level manifest from the workspace manifest
    ///
    /// Creates a minimal manifest containing only the dependencies needed by this crate.
    #[must_use]
    pub fn generate_crate_manifest(workspace: &Self, crate_name: &str) -> Option<Self> {
        let crate_config = workspace.crates.get(crate_name)?;

        let mut crate_manifest = Self {
            version: workspace.version,
            llvm: LlvmConfig::default(),
            crates: BTreeMap::new(),
            dependencies: BTreeMap::new(),
        };

        // Copy only the dependencies this crate needs
        for dep_name in &crate_config.dependencies {
            if let Some(dep_def) = workspace.dependencies.get(dep_name) {
                crate_manifest
                    .dependencies
                    .insert(dep_name.clone(), dep_def.clone());
            }
        }

        Some(crate_manifest)
    }

    /// Create a default manifest by parsing the embedded workspace pecos.toml
    ///
    /// The workspace pecos.toml is embedded at compile time, providing a single
    /// source of truth for dependency versions and configurations.
    ///
    /// # Panics
    ///
    /// Panics if the embedded `pecos.toml` cannot be parsed. This indicates a build
    /// error since the manifest is validated at compile time.
    #[must_use]
    pub fn default_pecos() -> Self {
        // Embed the workspace pecos.toml at compile time
        const WORKSPACE_MANIFEST: &str = include_str!("../../../pecos.toml");

        // Parse the embedded manifest
        toml::from_str(WORKSPACE_MANIFEST)
            .expect("Failed to parse embedded pecos.toml - this is a build error")
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Generate a default `pecos.toml` file
///
/// # Errors
///
/// Returns an error if the manifest cannot be serialized or written to `path`.
pub fn generate_manifest(path: &Path) -> Result<()> {
    let manifest = Manifest::default_pecos();
    manifest.save(path)?;
    println!("Generated {}", path.display());
    Ok(())
}

/// Sync result for a single crate
#[derive(Debug)]
pub struct SyncResult {
    pub crate_name: String,
    pub path: PathBuf,
    pub status: SyncStatus,
}

/// Status of a sync operation
#[derive(Debug)]
pub enum SyncStatus {
    /// Manifest was created (didn't exist before)
    Created,
    /// Manifest was updated (content changed)
    Updated,
    /// Manifest was already up to date
    UpToDate,
    /// Crate directory not found
    NotFound,
}

/// Sync crate manifests from workspace manifest
///
/// For each crate defined in the workspace manifest's `[crates.*]` section,
/// generates/updates a crate-level `pecos.toml` with just the dependencies
/// that crate needs.
///
/// Returns a list of results for each crate.
///
/// # Errors
///
/// Returns an error if the workspace manifest cannot be loaded or if any
/// crate manifest cannot be written.
pub fn sync_crate_manifests(workspace_path: &Path) -> Result<Vec<SyncResult>> {
    let workspace = Manifest::load(workspace_path)?;
    let workspace_dir = workspace_path
        .parent()
        .ok_or_else(|| Error::Config("Cannot determine workspace directory".into()))?;

    let mut results = Vec::new();

    for (crate_name, crate_config) in &workspace.crates {
        // Skip crates with no dependencies
        if crate_config.dependencies.is_empty() {
            continue;
        }

        // Find the crate directory
        let crate_dir = workspace_dir.join("crates").join(crate_name);
        let crate_manifest_path = crate_dir.join("pecos.toml");

        if !crate_dir.exists() {
            results.push(SyncResult {
                crate_name: crate_name.clone(),
                path: crate_manifest_path,
                status: SyncStatus::NotFound,
            });
            continue;
        }

        // Generate the crate manifest
        let Some(crate_manifest) = Manifest::generate_crate_manifest(&workspace, crate_name) else {
            continue;
        };

        // Check if manifest already exists and matches
        let status = if crate_manifest_path.exists() {
            let existing = Manifest::load(&crate_manifest_path)?;
            if manifests_match(&existing, &crate_manifest) {
                SyncStatus::UpToDate
            } else {
                SyncStatus::Updated
            }
        } else {
            SyncStatus::Created
        };

        // Write the manifest (if not already up to date)
        if !matches!(status, SyncStatus::UpToDate) {
            // Add a header comment
            let header = format!(
                "# PECOS dependency manifest for {crate_name}\n\
                 # This file is included in the published crate package\n\
                 # Generated by: cargo run -p pecos -- deps sync\n\n"
            );
            let content = toml::to_string_pretty(&crate_manifest)
                .map_err(|e| Error::Config(format!("Failed to serialize manifest: {e}")))?;
            fs::write(&crate_manifest_path, format!("{header}{content}"))?;
        }

        results.push(SyncResult {
            crate_name: crate_name.clone(),
            path: crate_manifest_path,
            status,
        });
    }

    Ok(results)
}

/// Check if two manifests have the same dependencies
fn manifests_match(a: &Manifest, b: &Manifest) -> bool {
    if a.dependencies.len() != b.dependencies.len() {
        return false;
    }

    for (name, dep_a) in &a.dependencies {
        match b.dependencies.get(name) {
            Some(dep_b) => {
                if dep_a.version != dep_b.version
                    || dep_a.url != dep_b.url
                    || dep_a.sha256 != dep_b.sha256
                {
                    return false;
                }
            }
            None => return false,
        }
    }

    true
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_round_trip() {
        let manifest = Manifest::default_pecos();
        let toml_str = toml::to_string_pretty(&manifest).unwrap();
        let parsed: Manifest = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.version, manifest.version);
        assert_eq!(parsed.dependencies.len(), manifest.dependencies.len());
        assert_eq!(parsed.crates.len(), manifest.crates.len());
    }

    #[test]
    fn test_crate_dependencies() {
        let manifest = Manifest::default_pecos();

        let quest_deps = manifest.get_crate_dependencies("pecos-quest");
        assert_eq!(quest_deps.len(), 1);

        let qulacs_deps = manifest.get_crate_dependencies("pecos-qulacs");
        assert_eq!(qulacs_deps.len(), 3); // qulacs, eigen, boost

        let ldpc_deps = manifest.get_crate_dependencies("pecos-ldpc-decoders");
        assert_eq!(ldpc_deps.len(), 3); // ldpc, stim, boost
    }

    #[test]
    fn test_llvm_requirements() {
        let manifest = Manifest::default_pecos();

        assert!(manifest.crate_requires_llvm("pecos-engines"));
        assert!(manifest.crate_requires_llvm("pecos"));
        assert!(!manifest.crate_requires_llvm("pecos-quest"));
    }

    #[test]
    fn test_dependency_users() {
        let manifest = Manifest::default_pecos();

        let stim_users = manifest.get_dependency_users("stim");
        assert!(stim_users.contains(&"pecos-ldpc-decoders"));

        let quest_users = manifest.get_dependency_users("quest");
        assert!(quest_users.contains(&"pecos-quest"));
    }
}
