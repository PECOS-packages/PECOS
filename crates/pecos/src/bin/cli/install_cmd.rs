//! Implementation of the unified `install` command

use pecos_build::Result;
use pecos_build::errors::Error;

/// Known installable targets
const KNOWN_TARGETS: &[&str] = &["cuda", "llvm", "cuquantum"];

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
        if !force && is_available(target) {
            println!(
                "[{}/{}] {target} is already available, skipping (use --force to install locally)",
                i + 1,
                total,
            );
            // Auto-configure LLVM if installed but not configured
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

/// Check if a target is already available on the system
fn is_available(target: &str) -> bool {
    match target {
        "cuda" => pecos_build::cuda::find_cuda().is_some(),
        "llvm" => pecos_build::llvm::find_llvm_14(None).is_some(),
        "cuquantum" => pecos_build::cuquantum::find_cuquantum().is_some(),
        _ => false,
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
