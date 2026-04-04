//! Implementation of the `setup` command
//!
//! Detects missing optional dependencies and interactively prompts the user
//! to install them. Designed to be called before `build` so that the build
//! environment is ready.

use pecos_build::Result;
use pecos_build::prompt::{PromptMode, confirm};

/// Run the setup command.
///
/// Checks for LLVM, CUDA, and cuQuantum and offers to install each one
/// that is missing. Prompt defaults follow the principle of least surprise:
///
/// - LLVM: default **yes** (required for full build, ~400 MB)
/// - CUDA: default **no** (large download ~4 GB, needs NVIDIA GPU)
/// - cuQuantum: default **yes** when CUDA is present (small, almost always wanted)
///
/// When `quiet` is true, suppresses output when all deps are already found.
/// Prompts and install output are still shown when something needs action.
pub fn run(mode: PromptMode, skip_llvm: bool, skip_cuda: bool, quiet: bool) -> Result<()> {
    // Check for legacy installs that should be migrated
    check_legacy_deps(mode)?;

    if !skip_llvm {
        setup_llvm(mode, quiet)?;
    }

    if !skip_cuda {
        setup_cuda(mode, quiet)?;
    }

    // cuQuantum: only relevant when CUDA is available
    if !skip_cuda && pecos_build::cuda::find_cuda().is_some() {
        setup_cuquantum(mode, quiet)?;
    }

    if !quiet {
        println!("Setup complete.");
    }
    Ok(())
}

// ── Migration ──────────────────────────────────────────────────────────────

fn check_legacy_deps(mode: PromptMode) -> Result<()> {
    let legacy = pecos_build::home::find_legacy_deps()?;
    if legacy.is_empty() {
        return Ok(());
    }

    // Always print migration prompts regardless of quiet flag
    println!("Found dependencies at legacy paths:");
    for dep in &legacy {
        println!("  {} -> {}", dep.old.display(), dep.new.display());
    }

    if confirm("Migrate to ~/.pecos/deps/?", true, mode) {
        for dep in &legacy {
            print!("  Moving {}...", dep.name);
            pecos_build::home::migrate_legacy_dep(dep)?;
            println!(" done");
        }
        println!();
    } else {
        println!("Skipping migration. Run `pecos migrate` later to move them.");
        println!();
    }

    Ok(())
}

// ── LLVM ────────────────────────────────────────────────────────────────────

fn setup_llvm(mode: PromptMode, quiet: bool) -> Result<()> {
    if pecos_build::llvm::find_llvm_14(None).is_some() {
        if !quiet {
            println!("LLVM 14: found");
        }
        ensure_llvm_configured(quiet);
        return Ok(());
    }

    if confirm(
        "LLVM 14 not found. Install to ~/.pecos/deps/llvm/ (~400 MB)?",
        true,
        mode,
    ) {
        pecos_build::llvm::installer::install_llvm(false, false)?;
    } else {
        println!("Skipping LLVM. QIR features will not be available.");
    }

    Ok(())
}

// ── CUDA ────────────────────────────────────────────────────────────────────

fn setup_cuda(mode: PromptMode, quiet: bool) -> Result<()> {
    if pecos_build::cuda::find_cuda().is_some() {
        if !quiet {
            println!("CUDA: found");
        }
        return Ok(());
    }

    // Only offer CUDA on platforms where it is supported
    if !cuda_platform_supported() {
        if !quiet {
            println!("CUDA: skipped (not supported on this platform)");
        }
        return Ok(());
    }

    if confirm(
        "CUDA not found. Install to ~/.pecos/deps/cuda/ (~4 GB)?",
        false,
        mode,
    ) {
        pecos_build::cuda::installer::install_cuda(false)?;
    } else {
        println!("Skipping CUDA. GPU features will not be available.");
    }

    Ok(())
}

// ── cuQuantum ───────────────────────────────────────────────────────────────

fn setup_cuquantum(mode: PromptMode, quiet: bool) -> Result<()> {
    if pecos_build::cuquantum::find_cuquantum().is_some() {
        if !quiet {
            println!("cuQuantum: found");
        }
        return Ok(());
    }

    if confirm(
        "cuQuantum not found. Install to ~/.pecos/deps/cuquantum/?",
        true,
        mode,
    ) {
        pecos_build::cuquantum::installer::install_cuquantum(false)?;
    } else {
        println!("Skipping cuQuantum.");
    }

    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Ensure LLVM is configured in `.cargo/config.toml` after detection/install.
fn ensure_llvm_configured(quiet: bool) {
    let config = pecos_build::llvm::config::validate_llvm_config();
    if config.is_healthy() {
        return;
    }
    if !quiet {
        println!("LLVM found but not configured, configuring...");
    }
    match pecos_build::llvm::config::auto_configure_llvm(None) {
        Ok(path) => {
            if !quiet {
                println!(
                    "Updated .cargo/config.toml with LLVM path: {}",
                    path.display()
                );
            }
        }
        Err(e) => {
            // Always show errors
            eprintln!("Warning: could not auto-configure LLVM: {e}");
            config.print_warnings();
        }
    }
}

/// Returns true if the current platform supports CUDA installation.
fn cuda_platform_supported() -> bool {
    cfg!(target_os = "linux") || cfg!(target_os = "windows")
}
