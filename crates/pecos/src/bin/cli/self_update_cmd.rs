//! Implementation of `pecos self` subcommands

use pecos_build::Result;
use pecos_build::errors::Error;
use std::process::Command;

/// Run `pecos self upgrade` — rebuilds and reinstalls the CLI from the repo.
pub fn run() -> Result<()> {
    let repo_root = pecos_build::llvm::find_cargo_project_root().ok_or_else(|| {
        Error::Config("Could not find PECOS repo root (no Cargo.toml found)".into())
    })?;

    let pecos_crate = repo_root.join("crates").join("pecos");
    if !pecos_crate.join("Cargo.toml").exists() {
        return Err(Error::Config(format!(
            "Expected pecos crate at {} but Cargo.toml not found",
            pecos_crate.display()
        )));
    }

    println!("Rebuilding pecos CLI from {}...", pecos_crate.display());
    println!();

    let status = Command::new("cargo")
        .args([
            "install",
            "--path",
            &pecos_crate.to_string_lossy(),
            "--features",
            "cli",
        ])
        .status()
        .map_err(|e| Error::Config(format!("Failed to run cargo install: {e}")))?;

    if !status.success() {
        return Err(Error::Config(format!(
            "cargo install failed with exit code: {}",
            status.code().unwrap_or(-1)
        )));
    }

    println!();
    println!("pecos CLI updated successfully.");
    Ok(())
}

/// Run `pecos self uninstall` — removes the pecos CLI binary.
pub fn run_uninstall() -> Result<()> {
    println!("Uninstalling pecos CLI...");
    println!();

    let status = Command::new("cargo")
        .args(["uninstall", "pecos"])
        .status()
        .map_err(|e| Error::Config(format!("Failed to run cargo uninstall: {e}")))?;

    if !status.success() {
        return Err(Error::Config(format!(
            "cargo uninstall failed with exit code: {}",
            status.code().unwrap_or(-1)
        )));
    }

    println!();
    println!("pecos CLI uninstalled successfully.");
    println!("To reinstall, run: cargo install --path crates/pecos --features cli");
    Ok(())
}
