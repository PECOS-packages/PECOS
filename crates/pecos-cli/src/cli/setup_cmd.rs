//! Implementation of the `setup` command
//!
//! Detects missing optional dependencies and interactively prompts the user
//! to install them. Designed to be the first thing a developer runs.

use pecos_build::Result;
use pecos_build::prompt::{PromptMode, confirm};
use std::fs;
use std::path::{Path, PathBuf};

/// Run the setup command.
///
/// Shows a summary of what's installed and what's missing, then offers
/// to install each missing dependency with Y/n prompts.
pub fn run(
    mode: PromptMode,
    skip_llvm: bool,
    skip_cuda: bool,
    skip_cmake: bool,
    quiet: bool,
) -> Result<()> {
    // Check for legacy installs that should be migrated
    check_legacy_deps(mode)?;

    // Remove stale Selene plugin scaffolding (file-less leftover directories)
    // that fail the workspace hygiene test. Quiet unless something is removed.
    sweep_stale_selene_plugins();

    let anything_missing = has_missing_deps(skip_llvm, skip_cuda, skip_cmake);

    // Show summary: always when not quiet, or when something needs action
    if !quiet || anything_missing {
        print_status_summary(skip_llvm, skip_cuda, skip_cmake);
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

    if !skip_cmake {
        setup_cmake(mode)?;
    }

    if !quiet || anything_missing {
        println!();
        println!("Setup complete. Run `just build` to build PECOS.");
    }
    Ok(())
}

fn has_missing_deps(skip_llvm: bool, skip_cuda: bool, skip_cmake: bool) -> bool {
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
    if !skip_cmake && pecos_build::cmake::find_cmake().is_none() {
        return true;
    }
    false
}

fn print_status_summary(skip_llvm: bool, skip_cuda: bool, skip_cmake: bool) {
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

    // cmake (optional, used for the MWPF decoder)
    if skip_cmake {
        println!("  cmake:      skipped (--skip-cmake)");
    } else if let Some(path) = pecos_build::cmake::find_cmake() {
        println!("  cmake:      {}", path.display());
    } else {
        println!("  cmake:      not found (optional, enables the MWPF decoder)");
    }
}

// ── Selene plugin hygiene ───────────────────────────────────────────────────

/// Silently remove stale `pecos-selene-*` plugin directories that contain
/// only empty subdirectories (no `Cargo.toml`, no `pyproject.toml`, no files
/// anywhere).
///
/// These leftovers fail the `test_selene_plugin_workspace_members_are_explicit_and_complete`
/// hygiene gate and confuse new developers. Real plugins (Cargo.toml +
/// pyproject.toml present) and work-in-progress dirs with any file content
/// are left untouched, so this is safe to run on every `pecos setup`.
fn sweep_stale_selene_plugins() {
    let Some(repo_root) = find_repo_root() else {
        return;
    };
    let selene_dir = repo_root.join("python").join("selene-plugins");
    if !selene_dir.is_dir() {
        return;
    }

    let Ok(read_dir) = fs::read_dir(&selene_dir) else {
        return;
    };

    let mut stale: Vec<PathBuf> = Vec::new();
    for entry in read_dir.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("pecos-selene-") || !path.is_dir() {
            continue;
        }
        // Real plugins are clearly identifiable; skip them.
        if path.join("Cargo.toml").is_file() && path.join("pyproject.toml").is_file() {
            continue;
        }
        // Preserve WIP: only remove if the tree has zero files anywhere.
        if tree_has_any_file(&path) {
            continue;
        }
        stale.push(path);
    }

    if stale.is_empty() {
        return;
    }

    for p in &stale {
        match fs::remove_dir_all(p) {
            Ok(()) => {
                let display = p.strip_prefix(&repo_root).unwrap_or(p);
                println!(
                    "  Removed stale Selene plugin scaffolding: {}",
                    display.display()
                );
            }
            Err(e) => {
                eprintln!("  Warning: failed to remove {}: {e}", p.display());
            }
        }
    }
}

/// Walk the directory tree and report whether any regular file (or symlink)
/// exists anywhere under `dir`. Empty directories don't count.
fn tree_has_any_file(dir: &Path) -> bool {
    let Ok(read_dir) = fs::read_dir(dir) else {
        // Unreadable — assume content to avoid accidental deletion.
        return true;
    };
    for entry in read_dir.flatten() {
        let Ok(ft) = entry.file_type() else {
            return true;
        };
        if ft.is_file() || ft.is_symlink() {
            return true;
        }
        if ft.is_dir() && tree_has_any_file(&entry.path()) {
            return true;
        }
    }
    false
}

/// Find the repository root by walking upward from CWD looking for the
/// workspace `Cargo.toml`. Returns `None` if `pecos setup` is invoked outside
/// a PECOS checkout (in which case there's nothing for us to sweep).
fn find_repo_root() -> Option<PathBuf> {
    let mut current = std::env::current_dir().ok()?;
    loop {
        let cargo_toml = current.join("Cargo.toml");
        if let Ok(content) = fs::read_to_string(&cargo_toml)
            && content.contains("[workspace]")
        {
            return Some(current);
        }
        if !current.pop() {
            return None;
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

// ── cmake (optional, MWPF decoder) ──────────────────────────────────────────

// cmake is optional, so install failures degrade gracefully (mwpf disabled)
// instead of aborting setup. Returns Result for symmetry with setup_llvm etc.
#[allow(clippy::unnecessary_wraps)]
fn setup_cmake(mode: PromptMode) -> Result<()> {
    if pecos_build::cmake::find_cmake().is_some() {
        return Ok(());
    }

    let docs_url = pecos_build::cmake::DOCS_URL;
    let version = pecos_build::cmake::CMAKE_VERSION;
    let prompt = format!(
        "Install cmake {version}? (~50MB download to ~/.pecos/deps/cmake-{version}/, \
         enables the optional MWPF decoder)"
    );
    if !confirm(&prompt, true, mode) {
        println!("  Skipping cmake. MWPF decoder will not be available.");
        println!("  To install manually later: pecos install cmake");
        println!("  Or install cmake system-wide: {docs_url}");
        return Ok(());
    }

    if let Err(e) = pecos_build::cmake::installer::install_cmake(false) {
        eprintln!("  Warning: cmake install failed: {e}");
        eprintln!("  See {docs_url} for manual install instructions.");
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
