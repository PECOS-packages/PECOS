//! Implementation of deps subcommands

#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::needless_pass_by_value)]

use crate::Result;
use crate::cli::DepsCommands;
use crate::download::download_cached;
use crate::manifest::{Manifest, SyncStatus, generate_manifest, sync_crate_manifests};
use std::path::PathBuf;

/// Run a deps subcommand
pub fn run(command: DepsCommands) -> Result<()> {
    match command {
        DepsCommands::Init { force } => run_init(force),
        DepsCommands::Status => run_status(),
        DepsCommands::Sync { dry_run } => run_sync(dry_run),
        DepsCommands::Verify { deps } => run_verify(deps),
    }
}

fn run_init(force: bool) -> Result<()> {
    let manifest_path = PathBuf::from("pecos.toml");

    if manifest_path.exists() && !force {
        eprintln!("pecos.toml already exists. Use --force to overwrite.");
        std::process::exit(1);
    }

    generate_manifest(&manifest_path)?;
    println!();
    println!("Created pecos.toml with default PECOS dependencies.");

    Ok(())
}

fn run_status() -> Result<()> {
    println!("Manifest Status");
    println!("===============");
    println!();

    // Check for pecos.toml
    if let Some(manifest_path) = Manifest::find() {
        println!("pecos.toml: {}", manifest_path.display());
        match Manifest::load(&manifest_path) {
            Ok(manifest) => {
                println!("  Version: {}", manifest.version);
                println!(
                    "  LLVM: version {} (required: {})",
                    manifest.llvm.version, manifest.llvm.required
                );
                if !manifest.llvm.required_by.is_empty() {
                    println!("    Required by: {}", manifest.llvm.required_by.join(", "));
                }
                println!();

                println!("  Crates ({}):", manifest.crates.len());
                for (crate_name, config) in &manifest.crates {
                    let deps = if config.dependencies.is_empty() {
                        "none".to_string()
                    } else {
                        config.dependencies.join(", ")
                    };
                    let llvm = if config.requires_llvm { " [LLVM]" } else { "" };
                    println!("    {crate_name}: {deps}{llvm}");
                }
                println!();

                println!("  Dependencies ({}):", manifest.dependencies.len());
                for (name, def) in &manifest.dependencies {
                    let version_short = if def.version.len() > 12 {
                        &def.version[..12]
                    } else {
                        &def.version
                    };
                    let desc = def.description.as_deref().unwrap_or("");
                    println!("    {name}: {version_short} - {desc}");
                }
            }
            Err(e) => {
                println!("  Error parsing: {e}");
            }
        }
    } else {
        println!("pecos.toml: not found");
        println!("  Run 'pecos deps init' to create one.");
    }

    Ok(())
}

fn run_sync(dry_run: bool) -> Result<()> {
    println!("Syncing crate manifests from workspace...");
    println!();

    // Find workspace manifest
    let workspace_path = Manifest::find().ok_or_else(|| {
        crate::errors::Error::Config(
            "pecos.toml not found. Run from the PECOS workspace directory.".into(),
        )
    })?;

    // Check this is actually a workspace manifest
    let content = std::fs::read_to_string(&workspace_path)?;
    if !content.contains("[crates.") {
        return Err(crate::errors::Error::Config(
            "Found pecos.toml but it doesn't appear to be a workspace manifest (no [crates.*] sections).".into(),
        ));
    }

    println!("Workspace manifest: {}", workspace_path.display());
    println!();

    if dry_run {
        println!("Dry run mode - no changes will be made");
        println!();
    }

    // For dry run, we need to manually check what would change
    if dry_run {
        let workspace = Manifest::load(&workspace_path)?;
        let workspace_dir = workspace_path.parent().unwrap();

        for (crate_name, crate_config) in &workspace.crates {
            if crate_config.dependencies.is_empty() {
                continue;
            }

            let crate_dir = workspace_dir.join("crates").join(crate_name);
            let crate_manifest_path = crate_dir.join("pecos.toml");

            if !crate_dir.exists() {
                println!("  [NOT FOUND] {crate_name}: crate directory not found");
                continue;
            }

            if crate_manifest_path.exists() {
                // Check if it would change
                let existing = Manifest::load(&crate_manifest_path)?;
                let generated = Manifest::generate_crate_manifest(&workspace, crate_name);

                if let Some(new_manifest) = generated {
                    let would_match = existing.dependencies.len()
                        == new_manifest.dependencies.len()
                        && existing.dependencies.iter().all(|(name, dep)| {
                            new_manifest
                                .dependencies
                                .get(name)
                                .is_some_and(|new_dep| {
                                    dep.version == new_dep.version
                                        && dep.url == new_dep.url
                                        && dep.sha256 == new_dep.sha256
                                })
                        });

                    if would_match {
                        println!("  [UP TO DATE] {crate_name}");
                    } else {
                        println!("  [WOULD UPDATE] {crate_name}");
                        // Show what would change
                        for (dep_name, dep) in &new_manifest.dependencies {
                            if let Some(existing_dep) = existing.dependencies.get(dep_name) {
                                if dep.version != existing_dep.version {
                                    println!(
                                        "    {dep_name}: {} -> {}",
                                        existing_dep.version, dep.version
                                    );
                                }
                            } else {
                                println!("    {dep_name}: (new)");
                            }
                        }
                    }
                }
            } else {
                println!("  [WOULD CREATE] {crate_name}");
            }
        }
    } else {
        // Actually sync
        let results = sync_crate_manifests(&workspace_path)?;

        let mut created = 0;
        let mut updated = 0;
        let mut up_to_date = 0;
        let mut not_found = 0;

        for result in &results {
            match result.status {
                SyncStatus::Created => {
                    println!(
                        "  [CREATED] {}: {}",
                        result.crate_name,
                        result.path.display()
                    );
                    created += 1;
                }
                SyncStatus::Updated => {
                    println!(
                        "  [UPDATED] {}: {}",
                        result.crate_name,
                        result.path.display()
                    );
                    updated += 1;
                }
                SyncStatus::UpToDate => {
                    println!("  [UP TO DATE] {}", result.crate_name);
                    up_to_date += 1;
                }
                SyncStatus::NotFound => {
                    println!(
                        "  [NOT FOUND] {}: crate directory not found",
                        result.crate_name
                    );
                    not_found += 1;
                }
            }
        }

        println!();
        println!(
            "Sync complete: {created} created, {updated} updated, {up_to_date} up to date, {not_found} not found"
        );
    }

    Ok(())
}

fn run_verify(deps_filter: Option<String>) -> Result<()> {
    println!("Verifying dependency checksums...");
    println!();

    let manifest_path = Manifest::find().ok_or_else(|| {
        crate::errors::Error::Config(
            "pecos.toml not found. Run 'pecos deps init' first.".into(),
        )
    })?;

    let manifest = Manifest::load(&manifest_path)?;

    // Filter dependencies if specified
    let deps_to_verify: Vec<&str> = if let Some(filter) = &deps_filter {
        filter.split(',').map(str::trim).collect()
    } else {
        manifest.dependencies.keys().map(String::as_str).collect()
    };

    let mut verified = 0;
    let mut failed = 0;

    for dep_name in deps_to_verify {
        if !manifest.dependencies.contains_key(dep_name) {
            println!("  [SKIP] {dep_name}: not found in manifest");
            continue;
        }

        print!("  Checking {dep_name}... ");

        match manifest.get_download_info(dep_name) {
            Ok(info) => {
                // download_cached verifies the SHA256
                match download_cached(&info) {
                    Ok(_) => {
                        println!("OK (SHA256 verified)");
                        verified += 1;
                    }
                    Err(e) => {
                        println!("FAILED: {e}");
                        failed += 1;
                    }
                }
            }
            Err(e) => {
                println!("FAILED: {e}");
                failed += 1;
            }
        }
    }

    println!();
    println!("Verification complete: {verified} OK, {failed} failed");

    if failed > 0 {
        std::process::exit(1);
    }

    Ok(())
}
