//! Implementation of the `list` command

#![allow(clippy::unnecessary_wraps)]

use pecos_build::Result;
use pecos_build::deps::list_dependencies;
use pecos_build::home::{get_cache_dir, get_deps_dir, get_llvm_dir};
use pecos_build::llvm::{find_llvm_14, get_llvm_version, get_repo_root_from_manifest};
use std::fs;

/// Run the list command
pub fn run(verbose: bool) -> Result<()> {
    println!("PECOS Dependencies");
    println!("==================");
    println!();

    // LLVM status
    println!("LLVM 14:");
    let repo_root = get_repo_root_from_manifest();
    if let Some(llvm_path) = find_llvm_14(repo_root) {
        print!("  Status: Installed at {}", llvm_path.display());
        if let Ok(version) = get_llvm_version(&llvm_path) {
            println!(" (version {version})");
        } else {
            println!();
        }
    } else {
        println!("  Status: Not found");
        println!("  Install with: pecos install llvm");
    }
    println!();

    // List available dependencies
    println!("Available Dependencies:");
    for dep in list_dependencies() {
        println!("  {}: {} - {}", dep.name, dep.version, dep.description);
    }
    println!();

    // List extracted sources and cached archives
    if verbose {
        println!("Extracted Sources (~/.pecos/deps/):");
        if let Ok(deps_dir) = get_deps_dir() {
            if deps_dir.exists() {
                let mut found = false;
                if let Ok(entries) = fs::read_dir(&deps_dir) {
                    for entry in entries.flatten() {
                        if entry.path().is_dir() {
                            println!("  {}", entry.file_name().to_string_lossy());
                            found = true;
                        }
                    }
                }
                if !found {
                    println!("  (none)");
                }
            } else {
                println!("  (deps directory not created yet)");
            }
        }
        println!();

        println!("Downloaded Archives (~/.pecos/cache/):");
        if let Ok(cache_dir) = get_cache_dir() {
            if cache_dir.exists() {
                let mut found = false;
                if let Ok(entries) = fs::read_dir(&cache_dir) {
                    for entry in entries.flatten() {
                        if entry.path().is_file() {
                            println!("  {}", entry.file_name().to_string_lossy());
                            found = true;
                        }
                    }
                }
                if !found {
                    println!("  (none)");
                }
            } else {
                println!("  (cache directory not created yet)");
            }
        }
        println!();

        println!("LLVM Directory:");
        if let Ok(llvm_dir) = get_llvm_dir() {
            if llvm_dir.exists() {
                println!("  {}", llvm_dir.display());
            } else {
                println!("  (not installed)");
            }
        }
    }

    Ok(())
}
