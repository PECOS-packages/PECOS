//! Implementation of the unified `upgrade` command
//!
//! Upgrade is a force reinstall of the target.

use pecos_build::Result;
use pecos_build::errors::Error;
use pecos_build::prompt::{PromptMode, confirm};

/// Known upgradeable targets
const KNOWN_TARGETS: &[&str] = &["cuda", "llvm", "cuquantum"];

/// Run the upgrade command
pub fn run(targets: &[String], all: bool, no_configure: bool, yes: bool) -> Result<()> {
    let targets: Vec<&str> = if all {
        KNOWN_TARGETS.to_vec()
    } else {
        // Validate all targets before upgrading any
        for target in targets {
            let name = target.to_lowercase();
            if !KNOWN_TARGETS.contains(&name.as_str()) {
                return Err(Error::Config(format!(
                    "Unknown upgrade target: '{target}'. Valid targets: {}",
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

    println!("This will force-reinstall:");
    for target in &targets {
        println!("  {target}");
    }
    println!();

    let mode = if yes {
        PromptMode::AcceptAll
    } else {
        PromptMode::Interactive
    };

    if !confirm("Continue?", false, mode) {
        println!("Cancelled.");
        return Ok(());
    }

    let total = targets.len();
    for (i, target) in targets.iter().enumerate() {
        println!("[{}/{}] Upgrading {target}...", i + 1, total);
        println!();
        upgrade_target(target, no_configure)?;
        println!();
    }

    println!("All done. Run `just build` to rebuild PECOS.");
    Ok(())
}

/// Upgrade a single target (force reinstall)
fn upgrade_target(target: &str, no_configure: bool) -> Result<()> {
    match target {
        "cuda" => {
            pecos_build::cuda::installer::install_cuda(true)?;
        }
        "llvm" => {
            pecos_build::llvm::installer::install_llvm(true, no_configure)?;
        }
        "cuquantum" => {
            pecos_build::cuquantum::installer::install_cuquantum(true)?;
        }
        _ => unreachable!("target was validated above"),
    }
    Ok(())
}
