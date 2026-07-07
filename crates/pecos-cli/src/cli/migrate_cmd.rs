//! Implementation of the `migrate` command
//!
//! Moves legacy top-level installs (LLVM, CUDA, cuQuantum) from `~/.pecos/`
//! into `~/.pecos/deps/` to match the new directory layout.

use pecos_build::Result;
use pecos_build::home::{find_legacy_dep_status, migrate_legacy_dep};
use pecos_build::prompt::{PromptMode, confirm};
use std::path::PathBuf;

fn find_project_root() -> Result<PathBuf> {
    let mut dir = std::env::current_dir()?;
    loop {
        if dir.join("Cargo.toml").exists() && dir.join(".cargo").exists() {
            return Ok(dir);
        }
        if !dir.pop() {
            return Ok(std::env::current_dir()?);
        }
    }
}

/// Run the migrate command.
pub fn run(mode: PromptMode) -> Result<()> {
    let legacy = find_legacy_dep_status()?;

    if !legacy.incompatible.is_empty() {
        println!("Found legacy dependencies that cannot be migrated safely:");
        for dep in &legacy.incompatible {
            println!("  {} at {}", dep.name, dep.old.display());
            println!("    {}", dep.reason);
            if dep.name == "LLVM" {
                println!("    This path will not be moved into ~/.pecos/deps/llvm-21.1/.");
                println!("    Remove it before installing/configuring LLVM 21.1.");
                println!("    Then install LLVM 21.1 with `pecos install llvm`, or configure");
                println!(
                    "    an existing LLVM 21.1 install with `pecos llvm configure /path/to/llvm`."
                );
            }
        }
        println!();

        for dep in &legacy.incompatible {
            if dep.name != "LLVM" {
                continue;
            }
            if confirm(
                &format!("Remove incompatible legacy LLVM at {}?", dep.old.display()),
                true,
                mode,
            ) {
                print!("  Removing old LLVM...");
                pecos_build::home::remove_incompatible_legacy_dep(dep)?;
                println!(" done");
            } else {
                println!(
                    "  Keeping old LLVM at {}. It will not be used as LLVM 21.1.",
                    dep.old.display()
                );
            }
        }
        println!();
    }

    if legacy.migratable.is_empty() {
        if legacy.incompatible.is_empty() {
            println!("Nothing to migrate. All dependencies are already under ~/.pecos/deps/.");
        } else {
            println!("No compatible legacy dependencies can be migrated automatically.");
        }
        return Ok(());
    }

    println!("Migrating legacy dependencies to ~/.pecos/deps/:");
    for dep in &legacy.migratable {
        print!(
            "  {} : {} -> {}",
            dep.name,
            dep.old.display(),
            dep.new.display()
        );
        migrate_legacy_dep(dep)?;
        println!(" ... done");
    }

    // Update .cargo/config.toml to point to the new paths
    let llvm_dir = pecos_build::home::get_llvm_dir_path()?;
    if pecos_build::llvm::is_valid_llvm(&llvm_dir)
        && let Ok(project_root) = find_project_root()
        && pecos_build::llvm::config::write_cargo_config(&project_root, &llvm_dir, true).is_ok()
    {
        println!(
            "Updated .cargo/config.toml with LLVM path: {}",
            llvm_dir.display()
        );
    }

    println!();
    println!("Migration complete.");

    Ok(())
}
