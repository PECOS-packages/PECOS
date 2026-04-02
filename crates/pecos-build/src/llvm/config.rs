//! LLVM configuration management for `.cargo/config.toml`

use crate::errors::{Error, Result};
use crate::llvm::{
    find_cargo_project_root, find_llvm_14, get_pecos_command, get_repo_root_from_manifest,
    is_valid_llvm_14,
};
use std::fs;
use std::path::{Path, PathBuf};

/// Result of validating the LLVM configuration
#[derive(Debug)]
pub struct ConfigValidation {
    /// Path configured in .cargo/config.toml (if any)
    pub configured_path: Option<PathBuf>,
    /// Whether the configured path exists
    pub path_exists: bool,
    /// Whether the configured path is valid LLVM 14
    pub path_is_valid_llvm14: bool,
    /// Path that `find_llvm_14` would return
    pub detected_path: Option<PathBuf>,
    /// Whether config matches detected LLVM
    pub config_matches_detected: bool,
}

impl ConfigValidation {
    /// Check if the configuration is healthy
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.configured_path.is_some() && self.path_exists && self.path_is_valid_llvm14
    }

    /// Print validation warnings if there are issues
    #[allow(clippy::collapsible_if)]
    pub fn print_warnings(&self) {
        let cmd = get_pecos_command();

        if let Some(ref configured) = self.configured_path {
            if !self.path_exists {
                eprintln!();
                eprintln!(
                    "Warning: .cargo/config.toml points to {} which doesn't exist",
                    configured.display()
                );
                eprintln!();
                eprintln!("To fix this:");
                eprintln!("  1. Install LLVM 14 for PECOS (recommended):");
                eprintln!("       {cmd} install llvm");
                if self.detected_path.is_some() {
                    eprintln!("  2. Or use the detected system LLVM:");
                    eprintln!("       {cmd} llvm configure");
                }
            } else if !self.path_is_valid_llvm14 {
                eprintln!();
                eprintln!(
                    "Warning: .cargo/config.toml points to {} which is not valid LLVM 14",
                    configured.display()
                );
                eprintln!();
                eprintln!("To fix this:");
                eprintln!("  1. Install LLVM 14 for PECOS (recommended):");
                eprintln!("       {cmd} install llvm");
                if self.detected_path.is_some() {
                    eprintln!("  2. Or use the detected system LLVM:");
                    eprintln!("       {cmd} llvm configure");
                }
            } else if !self.config_matches_detected {
                if let Some(ref detected) = self.detected_path {
                    // Only warn if detected path is different and also valid
                    // (e.g., they might have a preferred path configured)
                    if configured != detected {
                        eprintln!();
                        eprintln!(
                            "Note: .cargo/config.toml uses {} but {} was also detected",
                            configured.display(),
                            detected.display()
                        );
                    }
                }
            }
        } else if self.detected_path.is_some() {
            eprintln!();
            eprintln!("Warning: No LLVM configured in .cargo/config.toml");
            eprintln!();
            eprintln!("To fix this:");
            eprintln!("  1. Install LLVM 14 for PECOS (recommended):");
            eprintln!("       {cmd} install llvm");
            eprintln!("  2. Or use the detected system LLVM:");
            eprintln!("       {cmd} llvm configure");
        }
    }
}

/// Read the configured LLVM path from .cargo/config.toml
///
/// Handles both TOML formats:
///   `LLVM_SYS_140_PREFIX = "/path/to/llvm"`
///   `LLVM_SYS_140_PREFIX = { value = "/path/to/llvm", force = true }`
#[must_use]
pub fn read_configured_llvm_path() -> Option<PathBuf> {
    let project_root = find_cargo_project_root()?;
    let config_path = project_root.join(".cargo").join("config.toml");
    let content = fs::read_to_string(&config_path).ok()?;
    let table: toml::Table = content.parse().ok()?;

    let env = table.get("env")?;
    let entry = env.get("LLVM_SYS_140_PREFIX")?;

    // Simple string: LLVM_SYS_140_PREFIX = "/path"
    if let Some(s) = entry.as_str() {
        return Some(PathBuf::from(s));
    }

    // Inline table: LLVM_SYS_140_PREFIX = { value = "/path", force = true }
    if let Some(t) = entry.as_table()
        && let Some(v) = t.get("value").and_then(|v| v.as_str())
    {
        return Some(PathBuf::from(v));
    }

    None
}

/// Validate the current LLVM configuration
#[must_use]
pub fn validate_llvm_config() -> ConfigValidation {
    let configured_path = read_configured_llvm_path();
    let repo_root = get_repo_root_from_manifest();
    let detected_path = find_llvm_14(repo_root);

    let (path_exists, path_is_valid_llvm14) = if let Some(ref path) = configured_path {
        (path.exists(), is_valid_llvm_14(path))
    } else {
        (false, false)
    };

    let config_matches_detected = match (&configured_path, &detected_path) {
        (Some(configured), Some(detected)) => configured == detected,
        (None, None) => true,
        _ => false,
    };

    ConfigValidation {
        configured_path,
        path_exists,
        path_is_valid_llvm14,
        detected_path,
        config_matches_detected,
    }
}

/// Automatically configure LLVM for PECOS
///
/// This function determines the best LLVM 14 installation to use and writes
/// it to `.cargo/config.toml` with `force=true`.
///
/// Priority order:
/// 1. `~/.pecos/deps/llvm` (PECOS-managed LLVM, new path)
/// 2. `~/.pecos/llvm` (legacy path)
/// 3. `LLVM_SYS_140_PREFIX` environment variable
/// 4. System LLVM 14 (Homebrew, system paths, etc.)
///
/// # Errors
///
/// Returns an error if no suitable LLVM 14 installation could be found
pub fn auto_configure_llvm(project_root: Option<PathBuf>) -> Result<PathBuf> {
    // Priority 1 & 2: Check ~/.pecos/deps/llvm and legacy ~/.pecos/llvm
    let mut pecos_llvm_paths = Vec::new();
    if let Ok(deps_llvm) = crate::home::get_llvm_dir_path() {
        pecos_llvm_paths.push(deps_llvm);
    }
    if let Ok(legacy_llvm) = crate::home::get_legacy_llvm_dir_path() {
        pecos_llvm_paths.push(legacy_llvm);
    }
    #[cfg(target_os = "windows")]
    if let Some(home_dir) = dirs::home_dir() {
        pecos_llvm_paths.push(home_dir.join(".pecos").join("LLVM-14"));
    }

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
///
/// # Errors
///
/// Returns an error if the `.cargo` directory cannot be created or the config file
/// cannot be written.
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
