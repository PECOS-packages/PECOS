//! Implementation of the `setup` command
//!
//! Detects missing optional dependencies and interactively prompts the user
//! to install them. Designed to be the first thing a developer runs.

use pecos_build::Result;
use pecos_build::prompt::{PromptMode, confirm};

/// Run the setup command.
///
/// Shows a summary of what's installed and what's missing, then offers
/// to install each missing dependency with Y/n prompts.
pub fn run(mode: PromptMode, skip_llvm: bool, skip_cuda: bool, quiet: bool) -> Result<()> {
    // Check for legacy installs that should be migrated
    check_legacy_deps(mode)?;

    let anything_missing = has_missing_deps(skip_llvm, skip_cuda);

    // Show summary: always when not quiet, or when something needs action
    if !quiet || anything_missing {
        print_status_summary(skip_llvm, skip_cuda);
        println!();
    }

    if !skip_llvm {
        setup_llvm(mode)?;
    }

    if !skip_cuda {
        setup_cuda(mode)?;
    }

    // cuQuantum: only relevant when CUDA is available
    if !skip_cuda && pecos_build::cuda::find_cuda().is_some() {
        setup_cuquantum(mode)?;
    }

    if !quiet || anything_missing {
        println!();
        println!("Setup complete. Run `just build` to build PECOS.");
    }
    Ok(())
}

fn has_missing_deps(skip_llvm: bool, skip_cuda: bool) -> bool {
    if !skip_llvm && pecos_build::llvm::find_llvm_14(None).is_none() {
        return true;
    }
    if !skip_cuda && cuda_platform_supported() && pecos_build::cuda::find_cuda().is_none() {
        return true;
    }
    if !skip_cuda
        && pecos_build::cuda::find_cuda().is_some()
        && pecos_build::cuquantum::find_cuquantum().is_none()
    {
        return true;
    }
    false
}

fn print_status_summary(skip_llvm: bool, skip_cuda: bool) {
    println!("PECOS dependency status:");
    println!();

    // LLVM
    if skip_llvm {
        println!("  LLVM 14:    skipped (--skip-llvm)");
    } else if let Some(path) = pecos_build::llvm::find_llvm_14(None) {
        println!("  LLVM 14:    {}", path.display());
    } else {
        println!("  LLVM 14:    not found (~400 MB, required for QIR/HUGR compilation)");
    }

    // CUDA
    if skip_cuda {
        println!("  CUDA:       skipped (--skip-cuda)");
    } else if !cuda_platform_supported() {
        println!("  CUDA:       not supported on this platform");
    } else if let Some(path) = pecos_build::cuda::find_cuda() {
        println!("  CUDA:       {}", path.display());
    } else {
        println!("  CUDA:       not found (~4 GB, required for GPU simulation)");
    }

    // cuQuantum (only show if CUDA is present)
    if !skip_cuda && pecos_build::cuda::find_cuda().is_some() {
        if let Some(path) = pecos_build::cuquantum::find_cuquantum() {
            println!("  cuQuantum:  {}", path.display());
        } else {
            println!("  cuQuantum:  not found (~200 MB, GPU-accelerated quantum simulation)");
        }
    }
}

// ── Migration ──────────────────────────────────────────────────────────────

fn check_legacy_deps(mode: PromptMode) -> Result<()> {
    let legacy = pecos_build::home::find_legacy_deps()?;
    if legacy.is_empty() {
        return Ok(());
    }

    println!("Found dependencies at legacy paths:");
    for dep in &legacy {
        println!("  {} -> {}", dep.old.display(), dep.new.display());
    }

    if confirm(
        "Migrate to versioned paths under ~/.pecos/deps/?",
        true,
        mode,
    ) {
        for dep in &legacy {
            print!("  Moving {}...", dep.name);
            pecos_build::home::migrate_legacy_dep(dep)?;
            println!(" done");
        }
        println!();
    } else {
        println!("Skipping migration. Run `pecos migrate` later.");
        println!();
    }

    Ok(())
}

// ── LLVM ────────────────────────────────────────────────────────────────────

fn setup_llvm(mode: PromptMode) -> Result<()> {
    if pecos_build::llvm::find_llvm_14(None).is_some() {
        ensure_llvm_configured();
        return Ok(());
    }

    let version = pecos_build::home::LLVM_VERSION;
    if confirm(
        &format!("Install LLVM {version}? (~400 MB download, required for QIR/HUGR)"),
        true,
        mode,
    ) {
        pecos_build::llvm::installer::install_llvm(false, false)?;
    } else {
        println!("  Skipping LLVM. QIR/HUGR features will not be available.");
    }

    Ok(())
}

// ── CUDA ────────────────────────────────────────────────────────────────────

fn setup_cuda(mode: PromptMode) -> Result<()> {
    if pecos_build::cuda::find_cuda().is_some() {
        return Ok(());
    }

    if !cuda_platform_supported() {
        return Ok(());
    }

    if confirm(
        "Install CUDA Toolkit? (~4 GB download, requires NVIDIA GPU)",
        false, // default no -- it's big and optional
        mode,
    ) {
        pecos_build::cuda::installer::install_cuda(false)?;
    } else {
        println!("  Skipping CUDA. GPU simulation will not be available.");
    }

    Ok(())
}

// ── cuQuantum ───────────────────────────────────────────────────────────────

fn setup_cuquantum(mode: PromptMode) -> Result<()> {
    if pecos_build::cuquantum::find_cuquantum().is_some() {
        return Ok(());
    }

    if confirm(
        "Install cuQuantum? (~200 MB download, GPU-accelerated quantum simulation)",
        true, // default yes when CUDA is present
        mode,
    ) {
        pecos_build::cuquantum::installer::install_cuquantum(false)?;
    } else {
        println!("  Skipping cuQuantum.");
    }

    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn ensure_llvm_configured() {
    let config = pecos_build::llvm::config::validate_llvm_config();
    if config.is_healthy() {
        return;
    }
    println!("  LLVM found but not configured, configuring...");
    match pecos_build::llvm::config::auto_configure_llvm(None) {
        Ok(path) => println!("  Updated .cargo/config.toml: {}", path.display()),
        Err(e) => {
            eprintln!("  Warning: could not auto-configure LLVM: {e}");
            config.print_warnings();
        }
    }
}

fn cuda_platform_supported() -> bool {
    cfg!(target_os = "linux") || cfg!(target_os = "windows")
}
