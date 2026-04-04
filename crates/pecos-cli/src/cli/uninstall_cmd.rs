//! Implementation of the unified `uninstall` command

use pecos_build::Result;
use pecos_build::errors::Error;
use pecos_build::prompt::{PromptMode, confirm};
use std::path::PathBuf;

/// Known uninstallable targets
const KNOWN_TARGETS: &[&str] = &["cuda", "llvm", "cuquantum"];

/// Run the uninstall command
pub fn run(targets: &[String], all: bool, yes: bool) -> Result<()> {
    let targets: Vec<&str> = if all {
        KNOWN_TARGETS.to_vec()
    } else {
        // Validate all targets before uninstalling any
        for target in targets {
            let name = target.to_lowercase();
            if !KNOWN_TARGETS.contains(&name.as_str()) {
                return Err(Error::Config(format!(
                    "Unknown uninstall target: '{target}'. Valid targets: {}",
                    KNOWN_TARGETS.join(", ")
                )));
            }
        }
        // Deduplicate and order (reverse dependency order: cuquantum -> cuda -> llvm)
        let mut ordered = Vec::new();
        let lowered: Vec<String> = targets.iter().map(|t| t.to_lowercase()).collect();
        for &target in KNOWN_TARGETS.iter().rev() {
            if lowered.contains(&target.to_string()) && !ordered.contains(&target) {
                ordered.push(target);
            }
        }
        ordered
    };

    // Collect what will actually be removed (skip targets that aren't installed)
    let mut removals: Vec<(&str, PathBuf)> = Vec::new();
    for &target in &targets {
        if let Some(path) = installed_path(target) {
            removals.push((target, path));
        }
    }

    if removals.is_empty() {
        println!("Nothing to uninstall.");
        return Ok(());
    }

    // Show what will be removed and ask for confirmation
    println!("This will remove:");
    for (name, path) in &removals {
        let size = dir_size_display(path);
        println!("  {name}: {} ({size})", path.display());
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

    let total = removals.len();
    for (i, (target, _)) in removals.iter().enumerate() {
        println!("[{}/{}] Uninstalling {target}...", i + 1, total);
        uninstall_target(target)?;
        println!();
    }

    println!("All done.");
    Ok(())
}

/// Get the installed path for a target, or None if not locally installed.
fn installed_path(target: &str) -> Option<PathBuf> {
    let path = match target {
        "cuda" => pecos_build::home::get_cuda_dir_path().ok()?,
        "llvm" => pecos_build::home::get_llvm_dir_path().ok()?,
        "cuquantum" => pecos_build::home::get_cuquantum_dir_path().ok()?,
        _ => return None,
    };
    if path.exists() { Some(path) } else { None }
}

/// Get a human-readable size string for a directory.
fn dir_size_display(path: &PathBuf) -> String {
    match dir_size(path) {
        Some(bytes) if bytes >= 1_073_741_824 => {
            format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
        }
        Some(bytes) if bytes >= 1_048_576 => format!("{:.0} MB", bytes as f64 / 1_048_576.0),
        Some(bytes) => format!("{bytes} bytes"),
        None => "unknown size".to_string(),
    }
}

/// Recursively compute directory size in bytes.
fn dir_size(path: &PathBuf) -> Option<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path).ok()? {
        let entry = entry.ok()?;
        let meta = entry.metadata().ok()?;
        if meta.is_dir() {
            total += dir_size(&entry.path())?;
        } else {
            total += meta.len();
        }
    }
    Some(total)
}

/// Uninstall a single target
fn uninstall_target(target: &str) -> Result<()> {
    match target {
        "cuda" => pecos_build::cuda::installer::uninstall_cuda(),
        "llvm" => pecos_build::llvm::installer::uninstall_llvm(),
        "cuquantum" => pecos_build::cuquantum::installer::uninstall_cuquantum(),
        _ => unreachable!("target was validated above"),
    }
}
