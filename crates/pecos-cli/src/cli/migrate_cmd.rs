//! Implementation of the `migrate` command
//!
//! Moves legacy top-level installs (LLVM, CUDA, cuQuantum) from `~/.pecos/`
//! into `~/.pecos/deps/` to match the new directory layout.

use pecos_build::Result;
use pecos_build::home::{find_legacy_deps, migrate_legacy_dep};
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
pub fn run() -> Result<()> {
    let legacy = find_legacy_deps()?;

    if legacy.is_empty() {
        println!("Nothing to migrate. All dependencies are already under ~/.pecos/deps/.");
        return Ok(());
    }

    println!("Migrating legacy dependencies to ~/.pecos/deps/:");
    for dep in &legacy {
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
    if llvm_dir.exists()
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
