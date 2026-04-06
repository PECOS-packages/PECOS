//! Implementation of deps subcommands

use super::DepsCommands;
use pecos_build::Result;
use pecos_build::download::download_cached;
use pecos_build::manifest::{Manifest, check_consistency, collect_all_deps};
use std::path::PathBuf;

/// Run a deps subcommand
pub fn run(command: DepsCommands) -> Result<()> {
    match command {
        DepsCommands::Check => run_check(),
        DepsCommands::Status => {
            run_status();
            Ok(())
        }
        DepsCommands::Verify { deps } => run_verify(deps.as_deref()),
        DepsCommands::List => {
            run_list();
            Ok(())
        }
    }
}

fn run_list() {
    let workspace_root = find_workspace_root();
    let deps = if let Some(root) = workspace_root {
        collect_all_deps(&root).unwrap_or_default()
    } else {
        Manifest::find_or_default().dependencies
    };

    if deps.is_empty() {
        println!("No dependencies defined in any pecos.toml");
    } else {
        println!("Dependencies (from per-crate pecos.toml files):");
        println!();
        for (name, def) in &deps {
            let version_short = if def.version.len() > 12 {
                &def.version[..12]
            } else {
                &def.version
            };
            let desc = def.description.as_deref().unwrap_or("");
            println!("  {name:<20} {version_short} - {desc}");
        }
    }
}

fn run_check() -> Result<()> {
    let workspace_root = find_workspace_root().ok_or_else(|| {
        pecos_build::errors::Error::Config(
            "Cannot find workspace root (no crates/ directory found).".into(),
        )
    })?;

    println!("Checking dependency consistency across per-crate pecos.toml files...");
    println!();

    let mismatches = check_consistency(&workspace_root)?;

    if mismatches.is_empty() {
        println!("All shared dependencies are consistent.");
        Ok(())
    } else {
        println!("Mismatches found:");
        println!();
        for m in &mismatches {
            println!("  {} ({})", m.dep_name, m.field);
            for (crate_name, value) in &m.values {
                println!("    {crate_name}: {value}");
            }
            println!();
        }
        Err(pecos_build::errors::Error::Config(format!(
            "{} inconsistencies found. Edit per-crate pecos.toml files to resolve.",
            mismatches.len()
        )))
    }
}

fn run_status() {
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
    }
}

fn run_verify(deps_filter: Option<&str>) -> Result<()> {
    println!("Verifying dependency checksums...");
    println!();

    // Build a merged manifest from all per-crate manifests if in a workspace,
    // otherwise fall back to find()
    let manifest = match find_workspace_root() {
        Some(root) => {
            let deps = collect_all_deps(&root)?;
            Manifest {
                dependencies: deps,
                ..Manifest::default()
            }
        }
        None => Manifest::find_and_load()?,
    };

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

fn find_workspace_root() -> Option<PathBuf> {
    // Try CARGO_MANIFEST_DIR first, then current directory
    let start = std::env::var("CARGO_MANIFEST_DIR").map_or_else(
        |_| std::env::current_dir().unwrap_or_default(),
        PathBuf::from,
    );

    let mut path = start.as_path();
    loop {
        if path.join("crates").is_dir() && path.join("Cargo.toml").exists() {
            return Some(path.to_path_buf());
        }
        path = path.parent()?;
    }
}
