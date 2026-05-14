//! Implementation of the unified `install` command

use pecos_build::Result;
use pecos_build::errors::Error;
use pecos_build::prompt::{PromptMode, confirm};

/// Known installable targets
const KNOWN_TARGETS: &[&str] = &["cuda", "llvm", "cuquantum", "cmake"];

/// Run the install command
pub fn run(targets: &[String], force: bool, all: bool, no_configure: bool) -> Result<()> {
    let targets: Vec<&str> = if all {
        KNOWN_TARGETS.to_vec()
    } else {
        // Validate all targets before installing any
        for target in targets {
            let name = target.to_lowercase();
            if !KNOWN_TARGETS.contains(&name.as_str()) {
                return Err(Error::Config(format!(
                    "Unknown install target: '{target}'. Valid targets: {}",
                    KNOWN_TARGETS.join(", ")
                )));
            }
        }
        // Deduplicate and order by dependency: llvm -> cuda -> cuquantum
        let mut ordered = Vec::new();
        let lowered: Vec<String> = targets.iter().map(|t| t.to_lowercase()).collect();
        for &target in KNOWN_TARGETS {
            if lowered.contains(&target.to_string()) && !ordered.contains(&target) {
                ordered.push(target);
            }
        }
        ordered
    };

    let total = targets.len();
    for (i, target) in targets.iter().enumerate() {
        let existing = find_existing(target);
        let is_local = existing
            .as_ref()
            .is_some_and(|p| p.to_string_lossy().contains(".pecos/deps/"));

        if let Some(path) = existing.as_ref().filter(|_| !force) {
            if is_local {
                println!(
                    "[{}/{}] {target}: already installed at {}",
                    i + 1,
                    total,
                    path.display()
                );
            } else {
                println!(
                    "[{}/{}] {target}: found system install at {}",
                    i + 1,
                    total,
                    path.display()
                );
                if confirm(
                    "  Install a PECOS-managed copy to ~/.pecos/deps/ instead?",
                    false,
                    PromptMode::Interactive,
                ) {
                    println!();
                    install_target(target, true, no_configure)?;
                }
            }
            if *target == "llvm" {
                ensure_llvm_configured(no_configure);
            }
        } else {
            println!("[{}/{}] Installing {target}...", i + 1, total);
            println!();
            install_target(target, force, no_configure)?;
        }
        println!();
    }

    println!("All done. Run `just build` to build PECOS.");
    Ok(())
}

/// Find where a target is currently installed (if at all)
fn find_existing(target: &str) -> Option<std::path::PathBuf> {
    match target {
        "cuda" => pecos_build::cuda::find_cuda(),
        "llvm" => pecos_build::llvm::find_llvm_14(None),
        "cuquantum" => pecos_build::cuquantum::find_cuquantum(),
        "cmake" => pecos_build::cmake::find_cmake(),
        _ => None,
    }
}

/// Install a single target
fn install_target(target: &str, force: bool, no_configure: bool) -> Result<()> {
    match target {
        "cuda" => {
            pecos_build::cuda::installer::install_cuda(force)?;
        }
        "llvm" => {
            pecos_build::llvm::installer::install_llvm(force, no_configure)?;
        }
        "cuquantum" => {
            pecos_build::cuquantum::installer::install_cuquantum(force)?;
        }
        "cmake" => {
            pecos_build::cmake::installer::install_cmake(force)?;
        }
        _ => unreachable!("target was validated above"),
    }
    Ok(())
}

/// Ensure LLVM is configured in .cargo/config.toml when already installed.
/// Auto-configures if not healthy, unless --no-configure was passed.
fn ensure_llvm_configured(no_configure: bool) {
    let config = pecos_build::llvm::config::validate_llvm_config();
    if config.is_healthy() {
        return;
    }
    if no_configure {
        config.print_warnings();
        return;
    }
    println!("LLVM found but not configured, configuring...");
    match pecos_build::llvm::config::auto_configure_llvm(None) {
        Ok(path) => {
            println!(
                "Updated .cargo/config.toml with LLVM path: {}",
                path.display()
            );
        }
        Err(e) => {
            eprintln!("Warning: Could not auto-configure LLVM: {e}");
            config.print_warnings();
        }
    }
}
