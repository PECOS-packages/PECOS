//! Implementation of the `selene` subcommand

use pecos_build::Result;
use pecos_build::errors::Error;
use std::fs;
use std::path::{Path, PathBuf};

/// Selene plugin definition
struct SelenePlugin {
    /// Rust crate name (e.g., "pecos-selene-quest")
    crate_name: &'static str,
    /// Library base name without extension (e.g., `pecos_selene_quest`)
    lib_name: &'static str,
    /// Python package directory relative to repo root
    python_pkg_path: &'static str,
    /// Additional libraries to copy (e.g., CUDA backend)
    extra_libs: &'static [&'static str],
}

/// All known Selene plugins
const PLUGINS: &[SelenePlugin] = &[
    SelenePlugin {
        crate_name: "pecos-selene-quest",
        lib_name: "pecos_selene_quest",
        python_pkg_path: "python/selene-plugins/pecos-selene-quest/python/pecos_selene_quest",
        // CUDA backend library for GPU acceleration (built when --features cuda is used)
        extra_libs: &["pecos_quest_cuda"],
    },
    SelenePlugin {
        crate_name: "pecos-selene-qulacs",
        lib_name: "pecos_selene_qulacs",
        python_pkg_path: "python/selene-plugins/pecos-selene-qulacs/python/pecos_selene_qulacs",
        extra_libs: &[],
    },
    SelenePlugin {
        crate_name: "pecos-selene-sparsestab",
        lib_name: "pecos_selene_sparsestab",
        python_pkg_path: "python/selene-plugins/pecos-selene-sparsestab/python/pecos_selene_sparsestab",
        extra_libs: &[],
    },
    SelenePlugin {
        crate_name: "pecos-selene-statevec",
        lib_name: "pecos_selene_statevec",
        python_pkg_path: "python/selene-plugins/pecos-selene-statevec/python/pecos_selene_statevec",
        extra_libs: &[],
    },
];

/// Run the selene subcommand
pub fn run(command: super::SeleneCommands) -> Result<()> {
    match command {
        super::SeleneCommands::Install {
            plugin,
            profile,
            dry_run,
        } => run_install(plugin, &profile, dry_run),
        super::SeleneCommands::Clean {
            plugin,
            venv,
            dry_run,
            verbose,
        } => run_clean(plugin, venv, dry_run, verbose),
        super::SeleneCommands::List => run_list(),
    }
}

/// Get the repository root from the current directory
fn get_repo_root() -> Result<PathBuf> {
    // Try to find the repo root by looking for Cargo.toml with [workspace]
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
            return Err(Error::Selene(
                "Could not find PECOS repository root (no workspace Cargo.toml found)".to_string(),
            ));
        }
    }
}

/// Get the library filename for the current platform
fn get_lib_filename(lib_name: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{lib_name}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{lib_name}.dylib")
    } else {
        format!("lib{lib_name}.so")
    }
}

/// Get the target directory for a given profile
fn get_target_dir(repo_root: &Path, profile: &str) -> PathBuf {
    repo_root.join("target").join(profile)
}

/// Install Selene plugins by copying compiled libraries to Python package directories
#[allow(clippy::collapsible_if, clippy::too_many_lines)]
fn run_install(plugin: Option<String>, profile: &str, dry_run: bool) -> Result<()> {
    let repo_root = get_repo_root()?;
    let target_dir = get_target_dir(&repo_root, profile);

    // Filter plugins if a specific one was requested
    let plugins: Vec<&SelenePlugin> = match &plugin {
        Some(name) => PLUGINS
            .iter()
            .filter(|p| p.crate_name == name || p.lib_name == name.replace('-', "_"))
            .collect(),
        None => PLUGINS.iter().collect(),
    };

    if plugins.is_empty() {
        if let Some(name) = plugin {
            eprintln!("Unknown plugin: {name}");
            eprintln!("Available plugins:");
            for p in PLUGINS {
                eprintln!("  {}", p.crate_name);
            }
            return Err(Error::Selene(format!("Plugin '{name}' not found")));
        }
    }

    let mut installed = 0;
    let mut skipped = 0;
    let mut failed = 0;

    for p in plugins {
        let lib_filename = get_lib_filename(p.lib_name);
        let src = target_dir.join(&lib_filename);
        let dest_dir = repo_root.join(p.python_pkg_path).join("_dist/lib");
        let dest = dest_dir.join(&lib_filename);

        if !src.exists() {
            println!(
                "Skipping {}: library not built ({})",
                p.crate_name,
                src.display()
            );
            skipped += 1;
            continue;
        }

        if dry_run {
            println!("Would copy: {} -> {}", src.display(), dest.display());
            installed += 1;
            continue;
        }

        // Create destination directory
        if let Err(e) = fs::create_dir_all(&dest_dir) {
            eprintln!("Failed to create directory {}: {e}", dest_dir.display());
            failed += 1;
            continue;
        }

        // Copy the main library
        match fs::copy(&src, &dest) {
            Ok(bytes) => {
                println!(
                    "Installed {}: {} ({} bytes)",
                    p.crate_name,
                    dest.display(),
                    bytes
                );
                installed += 1;
            }
            Err(e) => {
                eprintln!(
                    "Failed to copy {} to {}: {e}",
                    src.display(),
                    dest.display()
                );
                failed += 1;
            }
        }

        // Copy extra libraries (e.g., CUDA backend) if they exist
        for extra_lib in p.extra_libs {
            let extra_filename = get_lib_filename(extra_lib);
            let extra_src = target_dir.join(&extra_filename);
            let extra_dest = dest_dir.join(&extra_filename);

            if extra_src.exists() {
                if dry_run {
                    println!(
                        "Would copy: {} -> {}",
                        extra_src.display(),
                        extra_dest.display()
                    );
                } else {
                    match fs::copy(&extra_src, &extra_dest) {
                        Ok(bytes) => {
                            println!(
                                "  + {}: {} ({} bytes)",
                                extra_lib,
                                extra_dest.display(),
                                bytes
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "  Warning: Failed to copy {} to {}: {e}",
                                extra_src.display(),
                                extra_dest.display()
                            );
                        }
                    }
                }
            }
        }
    }

    // Summary
    println!();
    if dry_run {
        println!("Dry run: {installed} would be installed, {skipped} skipped");
    } else {
        println!("Done: {installed} installed, {skipped} skipped, {failed} failed");
    }

    if failed > 0 {
        return Err(Error::Selene(format!(
            "{failed} plugin(s) failed to install"
        )));
    }

    Ok(())
}

/// Clean Selene plugin _dist directories and optionally venv installations
#[allow(clippy::collapsible_if)]
fn run_clean(plugin: Option<String>, venv: bool, dry_run: bool, verbose: u8) -> Result<()> {
    let repo_root = get_repo_root()?;

    // Filter plugins if a specific one was requested
    let plugins: Vec<&SelenePlugin> = match &plugin {
        Some(name) => PLUGINS
            .iter()
            .filter(|p| p.crate_name == name || p.lib_name == name.replace('-', "_"))
            .collect(),
        None => PLUGINS.iter().collect(),
    };

    if plugins.is_empty() {
        if let Some(name) = plugin {
            return Err(Error::Selene(format!("Plugin '{name}' not found")));
        }
    }

    let mut cleaned = 0;
    let mut skipped = 0;

    // Clean _dist directories
    for p in &plugins {
        let dist_dir = repo_root.join(p.python_pkg_path).join("_dist");

        if !dist_dir.exists() {
            skipped += 1;
            continue;
        }

        if dry_run {
            if verbose >= 1 {
                println!("Would remove: {}", dist_dir.display());
            }
            cleaned += 1;
            continue;
        }

        match fs::remove_dir_all(&dist_dir) {
            Ok(()) => {
                if verbose >= 1 {
                    println!("Removed: {}", dist_dir.display());
                }
                cleaned += 1;
            }
            Err(e) => {
                eprintln!("Failed to remove {}: {e}", dist_dir.display());
            }
        }
    }

    // Clean venv installations if requested
    if venv {
        cleaned += clean_venv_plugins(&repo_root, &plugins, dry_run, verbose);
    }

    // Summary (only if verbose or dry_run)
    if verbose >= 1 || dry_run {
        println!();
        if dry_run {
            println!("Dry run: {cleaned} would be cleaned, {skipped} already clean");
        } else {
            println!("Done: {cleaned} cleaned, {skipped} already clean");
        }
    }

    Ok(())
}

/// Clean selene plugins from .venv/lib/*/site-packages/
fn clean_venv_plugins(
    repo_root: &Path,
    plugins: &[&SelenePlugin],
    dry_run: bool,
    verbose: u8,
) -> usize {
    let venv_lib = repo_root.join(".venv/lib");
    if !venv_lib.exists() {
        return 0;
    }

    let mut cleaned = 0;

    // Find all python version directories
    if let Ok(entries) = fs::read_dir(&venv_lib) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("python") {
                    let site_packages = path.join("site-packages");
                    if site_packages.exists() {
                        cleaned += clean_site_packages(&site_packages, plugins, dry_run, verbose);
                    }
                }
            }
        }
    }

    cleaned
}

/// Clean selene plugins from a site-packages directory
fn clean_site_packages(
    site_packages: &Path,
    plugins: &[&SelenePlugin],
    dry_run: bool,
    verbose: u8,
) -> usize {
    let mut cleaned = 0;

    if let Ok(entries) = fs::read_dir(site_packages) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let path = entry.path();

            // Check if this matches any plugin
            for p in plugins {
                // Match package directory or dist-info directory
                if name == p.lib_name
                    || (name.starts_with(p.lib_name) && name.contains(".dist-info"))
                {
                    if dry_run {
                        if verbose >= 1 {
                            println!("Would remove: {}", path.display());
                        }
                        cleaned += 1;
                    } else if path.is_dir() {
                        if fs::remove_dir_all(&path).is_ok() {
                            if verbose >= 1 {
                                println!("Removed: {}", path.display());
                            }
                            cleaned += 1;
                        }
                    } else if fs::remove_file(&path).is_ok() {
                        if verbose >= 1 {
                            println!("Removed: {}", path.display());
                        }
                        cleaned += 1;
                    }
                    break;
                }
            }
        }
    }

    cleaned
}

/// List Selene plugins and their installation status
fn run_list() -> Result<()> {
    let repo_root = get_repo_root()?;

    println!("Selene Plugins:");
    println!();

    for p in PLUGINS {
        print!("  {}", p.crate_name);

        // Check if library is installed
        let dist_dir = repo_root.join(p.python_pkg_path).join("_dist/lib");
        let lib_filename = get_lib_filename(p.lib_name);
        let installed_lib = dist_dir.join(&lib_filename);

        if installed_lib.exists() {
            let size = installed_lib.metadata().map(|m| m.len()).unwrap_or(0);
            println!(" (installed, {size} bytes)");
        } else {
            println!(" (not installed)");
        }
    }

    // Check for available built libraries
    println!();
    println!("Built Libraries:");

    for profile in ["debug", "release", "native"] {
        let target_dir = get_target_dir(&repo_root, profile);
        let mut found = Vec::new();

        for p in PLUGINS {
            let lib_filename = get_lib_filename(p.lib_name);
            let lib_path = target_dir.join(&lib_filename);
            if lib_path.exists() {
                found.push(p.crate_name);
            }
        }

        if !found.is_empty() {
            println!("  {profile}: {}", found.join(", "));
        }
    }

    Ok(())
}
