//! Implementation of the `features` subcommand

use cargo_metadata::MetadataCommand;
use pecos_build::Result;
use pecos_build::errors::Error;
use std::collections::BTreeSet;

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

/// Get all features for a package using `cargo_metadata` crate
fn get_package_features(package: &str) -> Result<BTreeSet<String>> {
    let metadata = MetadataCommand::new()
        .no_deps()
        .exec()
        .map_err(|e| Error::Config(format!("Failed to get cargo metadata: {e}")))?;

    // Find the package in the workspace
    let pkg = metadata
        .packages
        .iter()
        .find(|p| p.name == package)
        .ok_or_else(|| Error::Config(format!("Package '{package}' not found in workspace")))?;

    // Extract feature names
    Ok(pkg.features.keys().cloned().collect())
}
