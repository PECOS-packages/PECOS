//! Implementation of the `features` subcommand

use crate::Result;
use crate::errors::Error;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Run the features subcommand
pub fn run(command: super::FeaturesCommands) -> Result<()> {
    match command {
        super::FeaturesCommands::List {
            package,
            exclude,
            json,
        } => run_list(&package, exclude.as_deref(), json),
    }
}

/// Get features for a package, optionally excluding some
fn run_list(package: &str, exclude: Option<&str>, json: bool) -> Result<()> {
    let features = get_package_features(package)?;

    // Parse exclusions
    let exclusions: BTreeSet<&str> = exclude
        .map(|e| e.split(',').map(str::trim).collect())
        .unwrap_or_default();

    // Filter features
    let filtered: Vec<&String> = features
        .iter()
        .filter(|f| !exclusions.contains(f.as_str()))
        .collect();

    if json {
        // Output as JSON array
        println!(
            "[{}]",
            filtered
                .iter()
                .map(|f| format!("\"{f}\""))
                .collect::<Vec<_>>()
                .join(", ")
        );
    } else {
        // Output as comma-separated list (for use in shell commands)
        println!(
            "{}",
            filtered
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(",")
        );
    }

    Ok(())
}

/// Get all features for a package using cargo metadata
fn get_package_features(package: &str) -> Result<BTreeSet<String>> {
    // Try cargo metadata first (most reliable)
    if let Ok(features) = get_features_from_cargo_metadata(package) {
        return Ok(features);
    }

    // Fall back to parsing Cargo.toml directly
    get_features_from_cargo_toml(package)
}

/// Get features using cargo metadata
fn get_features_from_cargo_metadata(package: &str) -> Result<BTreeSet<String>> {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .output()
        .map_err(|e| Error::Config(format!("Failed to run cargo metadata: {e}")))?;

    if !output.status.success() {
        return Err(Error::Config("cargo metadata failed".to_string()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Simple JSON parsing without external dependency
    // Look for the package and extract its features
    parse_features_from_metadata(&stdout, package)
}

/// Parse features from cargo metadata JSON output
#[allow(clippy::too_many_lines)]
fn parse_features_from_metadata(json: &str, package: &str) -> Result<BTreeSet<String>> {
    // Find the package in the JSON
    // Format: "packages": [{"name": "package-name", ..., "features": {"feat1": [...], "feat2": [...]}}]

    let mut features = BTreeSet::new();

    // First, find the "packages" array to search within
    let Some(packages_start) = json.find("\"packages\":[") else {
        return Err(Error::Config(
            "No packages array in cargo metadata".to_string(),
        ));
    };

    // Find the package section - look for "name":"package" pattern with proper boundary
    // The pattern should match "name":"package" followed by ", or "}
    let package_pattern = format!("\"name\":\"{package}\"");

    // Search within the packages array
    let packages_section = &json[packages_start..];

    // Find the correct occurrence by checking the character after the match
    let mut search_from = 0;
    let pkg_start_rel = loop {
        let Some(pos) = packages_section[search_from..].find(&package_pattern) else {
            return Err(Error::Config(format!(
                "Package '{package}' not found in cargo metadata"
            )));
        };
        let abs_pos = search_from + pos;
        let after_pos = abs_pos + package_pattern.len();

        // Check if this is an exact match (followed by , or } or whitespace)
        if after_pos < packages_section.len() {
            let next_char = packages_section.as_bytes()[after_pos] as char;
            if next_char == ',' || next_char == '}' || next_char.is_whitespace() {
                // Also verify this looks like a package (has "version": nearby, not "req":)
                let context_end = (abs_pos + 200).min(packages_section.len());
                let context = &packages_section[abs_pos..context_end];
                // Package definitions have "version": near the start, deps have "req":
                if context.contains("\"version\":")
                    && !context[..50.min(context.len())].contains("\"req\":")
                {
                    break abs_pos;
                }
            }
        }

        // Not the right match, continue searching
        search_from = abs_pos + 1;
    };

    let pkg_start = packages_start + pkg_start_rel;

    // Go backwards from pkg_start to find the opening brace of this package object
    // The package object starts with '{' before the "name" field
    let json_before = &json[..pkg_start];
    let Some(obj_start) = json_before.rfind('{') else {
        return Err(Error::Config("Malformed JSON".to_string()));
    };

    // Find the end of this package object by counting braces
    let package_obj = &json[obj_start..];
    let mut depth = 0;
    let mut obj_end = 0;
    for (i, c) in package_obj.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    obj_end = i + 1;
                    break;
                }
            }
            _ => {}
        }
    }

    let package_json = &package_obj[..obj_end];

    // Find the features object within this package
    let Some(features_start) = package_json.find("\"features\":{") else {
        // Package has no features
        return Ok(features);
    };

    // Extract the features object content
    let features_section = &package_json[features_start + 12..]; // Skip "\"features\":{"

    // Find matching closing brace
    let mut depth = 1;
    let mut end_idx = 0;
    for (i, c) in features_section.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end_idx = i;
                    break;
                }
            }
            _ => {}
        }
    }

    let features_content = &features_section[..end_idx];

    // Extract feature names - they're the keys in this JSON object
    // Pattern: "feature_name":[...] or "feature_name":[]
    // We just need to find strings followed by ":"
    let mut in_string = false;
    let mut escape_next = false;
    let mut current_key = String::new();
    let mut bracket_depth = 0;

    let chars: Vec<char> = features_content.chars().collect();

    for (i, &c) in chars.iter().enumerate() {
        if escape_next {
            if in_string {
                current_key.push(c);
            }
            escape_next = false;
            continue;
        }

        match c {
            '\\' => {
                escape_next = true;
            }
            '"' if !in_string => {
                // Only start a new key if we're at bracket depth 0 (not inside an array value)
                if bracket_depth == 0 {
                    in_string = true;
                    current_key.clear();
                }
            }
            '"' if in_string => {
                in_string = false;
                // Check if next non-whitespace is ':' (meaning this is a key)
                if bracket_depth == 0 {
                    let mut j = i + 1;
                    while j < chars.len() && chars[j].is_whitespace() {
                        j += 1;
                    }
                    if j < chars.len() && chars[j] == ':' {
                        // This is a feature name
                        features.insert(current_key.clone());
                    }
                }
            }
            '[' => bracket_depth += 1,
            ']' => bracket_depth -= 1,
            c if in_string => {
                current_key.push(c);
            }
            _ => {}
        }
    }

    Ok(features)
}

/// Get features by parsing Cargo.toml directly
fn get_features_from_cargo_toml(package: &str) -> Result<BTreeSet<String>> {
    let repo_root = get_repo_root()?;

    // Try to find the package's Cargo.toml
    let cargo_toml_path = find_package_cargo_toml(&repo_root, package)?;
    let content = fs::read_to_string(&cargo_toml_path)
        .map_err(|e| Error::Config(format!("Failed to read {}: {e}", cargo_toml_path.display())))?;

    Ok(parse_features_from_toml(&content))
}

/// Find a package's Cargo.toml
fn find_package_cargo_toml(repo_root: &Path, package: &str) -> Result<PathBuf> {
    // Common locations to check
    let candidates = [
        repo_root.join(format!("crates/{package}/Cargo.toml")),
        repo_root.join(format!("crates/{}/Cargo.toml", package.replace('-', "_"))),
        repo_root.join("Cargo.toml"), // Workspace root for workspace packages
    ];

    for path in &candidates {
        if path.exists() {
            let content = fs::read_to_string(path).ok();
            if let Some(content) = content {
                // Check if this Cargo.toml contains the package
                if content.contains(&format!("name = \"{package}\""))
                    || content.contains(&format!("name = '{package}'"))
                {
                    return Ok(path.clone());
                }
            }
        }
    }

    Err(Error::Config(format!(
        "Could not find Cargo.toml for package '{package}'"
    )))
}

/// Parse features from Cargo.toml content
fn parse_features_from_toml(content: &str) -> BTreeSet<String> {
    let mut features = BTreeSet::new();
    let mut in_features_section = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Check for [features] section start
        if trimmed == "[features]" {
            in_features_section = true;
            continue;
        }

        // Check for new section start (end of features)
        if trimmed.starts_with('[') && in_features_section {
            break;
        }

        // Parse feature definitions in [features] section
        if in_features_section && let Some(eq_pos) = trimmed.find('=') {
            let feature_name = trimmed[..eq_pos].trim();
            if !feature_name.is_empty() && !feature_name.starts_with('#') {
                features.insert(feature_name.to_string());
            }
        }
    }

    features
}

/// Get the repository root from the current directory
fn get_repo_root() -> Result<PathBuf> {
    let mut current = std::env::current_dir()?;

    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            let content = fs::read_to_string(&cargo_toml)?;
            if content.contains("[workspace]") {
                return Ok(current);
            }
        }

        if !current.pop() {
            return Err(Error::Config(
                "Could not find PECOS repository root".to_string(),
            ));
        }
    }
}
