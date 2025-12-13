//! LLVM configuration management for `.cargo/config.toml`

#![allow(clippy::missing_errors_doc)]

use crate::errors::{Error, Result};
use crate::llvm::{
    find_cargo_project_root, find_llvm_14, get_repo_root_from_manifest, is_valid_llvm_14,
};
use std::fs;
use std::path::{Path, PathBuf};

/// Automatically configure LLVM for PECOS
///
/// This function determines the best LLVM 14 installation to use and writes
/// it to `.cargo/config.toml` with `force=true`.
///
/// Priority order:
/// 1. `~/.pecos/llvm` (PECOS-managed LLVM)
/// 2. `LLVM_SYS_140_PREFIX` environment variable
/// 3. System LLVM 14 (Homebrew, system paths, etc.)
///
/// # Errors
///
/// Returns an error if no suitable LLVM 14 installation could be found
pub fn auto_configure_llvm(project_root: Option<PathBuf>) -> Result<PathBuf> {
    // Priority 1: Check ~/.pecos/ for PECOS-managed LLVM
    if let Some(home_dir) = dirs::home_dir() {
        let pecos_dir = home_dir.join(".pecos");

        #[cfg(target_os = "windows")]
        let pecos_llvm_paths = vec![pecos_dir.join("LLVM-14"), pecos_dir.join("llvm")];

        #[cfg(not(target_os = "windows"))]
        let pecos_llvm_paths = vec![pecos_dir.join("llvm")];

        for pecos_llvm in pecos_llvm_paths {
            if is_valid_llvm_14(&pecos_llvm) {
                let project_root = project_root
                    .or_else(get_repo_root_from_manifest)
                    .or_else(find_cargo_project_root)
                    .ok_or_else(|| Error::Config("Could not find Cargo project root".into()))?;

                write_cargo_config(&project_root, &pecos_llvm, true)?;
                return Ok(pecos_llvm);
            }
        }
    }

    // Priority 2: Check LLVM_SYS_140_PREFIX
    if let Ok(sys_prefix) = std::env::var("LLVM_SYS_140_PREFIX") {
        let path = PathBuf::from(&sys_prefix);
        if is_valid_llvm_14(&path) {
            let project_root = project_root
                .or_else(get_repo_root_from_manifest)
                .or_else(find_cargo_project_root)
                .ok_or_else(|| Error::Config("Could not find Cargo project root".into()))?;

            write_cargo_config(&project_root, &path, true)?;
            return Ok(path);
        }
    }

    // Priority 3: Scan system for LLVM 14
    let repo_root = get_repo_root_from_manifest();
    if let Some(detected_path) = find_llvm_14(repo_root) {
        let project_root = project_root
            .or_else(get_repo_root_from_manifest)
            .or_else(find_cargo_project_root)
            .ok_or_else(|| Error::Config("Could not find Cargo project root".into()))?;

        write_cargo_config(&project_root, &detected_path, true)?;
        return Ok(detected_path);
    }

    Err(Error::Llvm("No suitable LLVM 14 installation found".into()))
}

/// Write or update `.cargo/config.toml` with LLVM configuration
///
/// # Arguments
/// * `project_root` - Path to the Cargo project root
/// * `llvm_path` - Path to the LLVM installation
/// * `force` - If true, use `force=true` to override shell environment variables
pub fn write_cargo_config(project_root: &Path, llvm_path: &Path, force: bool) -> Result<()> {
    let cargo_dir = project_root.join(".cargo");
    let config_path = cargo_dir.join("config.toml");

    fs::create_dir_all(&cargo_dir)?;

    // Convert path to forward slashes for TOML compatibility
    let llvm_path_str = llvm_path.to_string_lossy().replace('\\', "/");

    let llvm_line = if force {
        format!("LLVM_SYS_140_PREFIX = {{ value = \"{llvm_path_str}\", force = true }}")
    } else {
        format!("LLVM_SYS_140_PREFIX = \"{llvm_path_str}\"")
    };

    let existing_content = fs::read_to_string(&config_path).unwrap_or_default();

    // Check if config already has correct LLVM_SYS_140_PREFIX
    if existing_content.contains("LLVM_SYS_140_PREFIX") {
        let simple_format = format!("LLVM_SYS_140_PREFIX = \"{llvm_path_str}\"");
        let force_format =
            format!("LLVM_SYS_140_PREFIX = {{ value = \"{llvm_path_str}\", force = true }}");

        if (force && existing_content.contains(&force_format))
            || (!force && existing_content.contains(&simple_format))
        {
            return Ok(());
        }

        // Update existing configuration
        let lines: Vec<&str> = existing_content.lines().collect();
        let mut new_lines = Vec::new();
        let mut in_env_section = false;
        let mut updated = false;
        let mut skip_next_lines = 0;

        for (i, line) in lines.iter().enumerate() {
            if skip_next_lines > 0 {
                skip_next_lines -= 1;
                continue;
            }

            let trimmed = line.trim();

            if trimmed.starts_with('[') {
                in_env_section = trimmed == "[env]";
            }

            if in_env_section && trimmed.starts_with("LLVM_SYS_140_PREFIX") {
                new_lines.push(llvm_line.clone());
                updated = true;

                if trimmed.contains('{') && !trimmed.contains('}') {
                    for line in lines.iter().skip(i + 1) {
                        skip_next_lines += 1;
                        if line.contains('}') {
                            break;
                        }
                    }
                }
            } else {
                new_lines.push((*line).to_string());
            }
        }

        if updated {
            fs::write(&config_path, new_lines.join("\n"))?;
            return Ok(());
        }
    }

    // No LLVM configuration exists, append it
    let llvm_config = format!(
        "\n# LLVM configuration for PECOS\n\
         [env]\n\
         {llvm_line}\n"
    );

    let new_content = if existing_content.is_empty() {
        llvm_config.trim_start().to_string()
    } else {
        format!("{existing_content}{llvm_config}")
    };

    fs::write(&config_path, new_content)?;
    Ok(())
}
