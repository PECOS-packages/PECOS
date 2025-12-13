//! External dependency definitions
//!
//! This module provides dependency information by reading from pecos.toml.
//! The workspace pecos.toml is embedded at compile time as a fallback.

use crate::manifest::Manifest;

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
