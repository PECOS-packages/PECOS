//! Implementation of the `clean` command
//!
//! Removes cached downloads and temporary files from `~/.pecos/`.

use pecos_build::Result;
use pecos_build::errors::Error;
use pecos_build::prompt::{PromptMode, confirm};
use std::fs;
use std::path::Path;

/// Known clean targets
const KNOWN_TARGETS: &[&str] = &["cache", "tmp"];

/// Run the clean command.
pub fn run(targets: &[String], all: bool, dry_run: bool, yes: bool) -> Result<()> {
    let targets: Vec<&str> = if all {
        KNOWN_TARGETS.to_vec()
    } else {
        for target in targets {
            let name = target.to_lowercase();
            if !KNOWN_TARGETS.contains(&name.as_str()) {
                return Err(Error::Config(format!(
                    "Unknown clean target: '{target}'. Valid targets: {}",
                    KNOWN_TARGETS.join(", ")
                )));
            }
        }
        targets.iter().map(String::as_str).collect()
    };

    let mut total_bytes = 0u64;
    let mut dirs_to_clean: Vec<(&str, std::path::PathBuf, u64)> = Vec::new();

    for &target in &targets {
        let dir = match target {
            "cache" => pecos_build::home::get_cache_dir_path()?,
            "tmp" => pecos_build::home::get_tmp_dir_path()?,
            _ => unreachable!(),
        };
        if dir.exists() {
            let size = dir_size(&dir).unwrap_or(0);
            if size > 0 {
                dirs_to_clean.push((target, dir, size));
                total_bytes += size;
            }
        }
    }

    if dirs_to_clean.is_empty() {
        println!("Nothing to clean.");
        return Ok(());
    }

    println!("Will remove:");
    for (name, path, size) in &dirs_to_clean {
        println!("  {name}: {} ({})", path.display(), format_bytes(*size));
    }
    println!("  Total: {}", format_bytes(total_bytes));
    println!();

    if dry_run {
        println!("Dry run -- no files removed.");
        return Ok(());
    }

    let mode = if yes {
        PromptMode::AcceptAll
    } else {
        PromptMode::Interactive
    };

    if !confirm("Continue?", false, mode) {
        println!("Cancelled.");
        return Ok(());
    }

    for (name, path, _) in &dirs_to_clean {
        print!("Cleaning {name}...");
        // Remove contents but keep the directory itself
        for entry in fs::read_dir(path)?.flatten() {
            if entry.path().is_dir() {
                fs::remove_dir_all(entry.path())?;
            } else {
                fs::remove_file(entry.path())?;
            }
        }
        println!(" done");
    }

    println!("Cleaned {}.", format_bytes(total_bytes));
    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.0} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn dir_size(path: &Path) -> Option<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(path).ok()? {
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
