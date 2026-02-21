//! Implementation of the unified `uninstall` command

use pecos_build::Result;
use pecos_build::errors::Error;

/// Known uninstallable targets
const KNOWN_TARGETS: &[&str] = &["cuda", "llvm", "cuquantum"];

/// Run the uninstall command
pub fn run(targets: &[String], all: bool) -> Result<()> {
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

    let total = targets.len();
    for (i, target) in targets.iter().enumerate() {
        println!("[{}/{}] Uninstalling {target}...", i + 1, total);
        uninstall_target(target)?;
        println!();
    }

    println!("All done.");
    Ok(())
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
