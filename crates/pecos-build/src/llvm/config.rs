//! LLVM configuration management for `.cargo/config.toml`

use crate::errors::{Error, Result};
use crate::llvm::{
    LLVM_SYS_PREFIX_ENV, REQUIRED_VERSION, find_cargo_project_root, find_llvm, get_pecos_command,
    get_repo_root_from_manifest, is_valid_llvm, normalize_path_string, path_to_env_string,
};
use std::fs;
use std::path::{Path, PathBuf};

/// Result of validating the LLVM configuration
#[derive(Debug)]
pub struct ConfigValidation {
    /// Path configured in .cargo/config.toml (if any)
    pub configured_path: Option<PathBuf>,
    /// Stale versioned LLVM env keys that should not be present.
    pub stale_llvm_prefix_envs: Vec<String>,
    /// Whether the configured path exists
    pub path_exists: bool,
    /// Whether the configured path is the required LLVM version
    pub path_is_valid_llvm: bool,
    /// Path that `find_llvm` would return
    pub detected_path: Option<PathBuf>,
    /// Whether config matches detected LLVM
    pub config_matches_detected: bool,
}

impl ConfigValidation {
    /// Check if the configuration is healthy
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.configured_path.is_some()
            && self.stale_llvm_prefix_envs.is_empty()
            && self.path_exists
            && self.path_is_valid_llvm
    }

    /// Print validation warnings if there are issues
    #[allow(clippy::collapsible_if)]
    pub fn print_warnings(&self) {
        let cmd = get_pecos_command();

        if let Some(ref configured) = self.configured_path {
            if !self.stale_llvm_prefix_envs.is_empty() {
                eprintln!();
                eprintln!(
                    "Warning: .cargo/config.toml contains stale LLVM config: {}",
                    self.stale_llvm_prefix_envs.join(", ")
                );
                eprintln!();
                eprintln!("To fix this:");
                eprintln!("  {cmd} llvm configure");
            }

            if !self.path_exists {
                eprintln!();
                eprintln!(
                    "Warning: .cargo/config.toml points to {} which doesn't exist",
                    configured.display()
                );
                eprintln!();
                eprintln!("To fix this:");
                eprintln!("  1. Install LLVM {REQUIRED_VERSION} for PECOS (recommended):");
                eprintln!("       {cmd} install llvm");
                if self.detected_path.is_some() {
                    eprintln!("  2. Or use the detected system LLVM:");
                    eprintln!("       {cmd} llvm configure");
                }
            } else if !self.path_is_valid_llvm {
                eprintln!();
                eprintln!(
                    "Warning: .cargo/config.toml points to {} which is not valid LLVM {REQUIRED_VERSION}",
                    configured.display()
                );
                eprintln!();
                eprintln!("To fix this:");
                eprintln!("  1. Install LLVM {REQUIRED_VERSION} for PECOS (recommended):");
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
            eprintln!("  1. Install LLVM {REQUIRED_VERSION} for PECOS (recommended):");
            eprintln!("       {cmd} install llvm");
            eprintln!("  2. Or use the detected system LLVM:");
            eprintln!("       {cmd} llvm configure");
        }
    }
}

/// Read the configured LLVM path from .cargo/config.toml
///
/// Handles both TOML formats:
///   `LLVM_SYS_211_PREFIX = "/path/to/llvm"`
///   `LLVM_SYS_211_PREFIX = { value = "/path/to/llvm", force = true }`
#[must_use]
pub fn read_configured_llvm_path() -> Option<PathBuf> {
    let project_root = find_cargo_project_root()?;
    let config_path = project_root.join(".cargo").join("config.toml");
    let content = fs::read_to_string(&config_path).ok()?;
    let table: toml::Table = content.parse().ok()?;

    let env = table.get("env")?;
    let entry = env.get(LLVM_SYS_PREFIX_ENV)?;

    // Simple string: LLVM_SYS_211_PREFIX = "/path"
    if let Some(s) = entry.as_str() {
        return Some(PathBuf::from(normalize_path_string(s)));
    }

    // Inline table: LLVM_SYS_211_PREFIX = { value = "/path", force = true }
    if let Some(t) = entry.as_table()
        && let Some(v) = t.get("value").and_then(|v| v.as_str())
    {
        return Some(PathBuf::from(normalize_path_string(v)));
    }

    None
}

fn is_versioned_llvm_prefix_env(key: &str) -> bool {
    let Some(version) = key
        .strip_prefix("LLVM_SYS_")
        .and_then(|rest| rest.strip_suffix("_PREFIX"))
    else {
        return false;
    };
    !version.is_empty() && version.bytes().all(|b| b.is_ascii_digit())
}

fn is_stale_llvm_prefix_env(key: &str) -> bool {
    is_versioned_llvm_prefix_env(key) && key != LLVM_SYS_PREFIX_ENV
}

/// Read stale versioned LLVM prefix keys from `.cargo/config.toml`.
///
/// PECOS uses `LLVM_SYS_211_PREFIX`. Older entries such as
/// `LLVM_SYS_140_PREFIX` can still affect transitive build scripts when Cargo
/// exports forced `[env]` values, so they should be removed.
#[must_use]
pub fn read_stale_llvm_prefix_envs(project_root: &Path) -> Vec<String> {
    let config_path = project_root.join(".cargo").join("config.toml");
    let Ok(content) = fs::read_to_string(&config_path) else {
        return Vec::new();
    };
    let Ok(table) = content.parse::<toml::Table>() else {
        return Vec::new();
    };
    let Some(env) = table.get("env").and_then(|env| env.as_table()) else {
        return Vec::new();
    };

    let mut stale: Vec<String> = env
        .keys()
        .filter(|key| is_stale_llvm_prefix_env(key))
        .cloned()
        .collect();
    stale.sort();
    stale
}

/// Validate the current LLVM configuration
#[must_use]
pub fn validate_llvm_config() -> ConfigValidation {
    let project_root = find_cargo_project_root();
    let configured_path = read_configured_llvm_path();
    let repo_root = get_repo_root_from_manifest();
    let detected_path = find_llvm(repo_root);
    let stale_llvm_prefix_envs = project_root
        .as_deref()
        .map(read_stale_llvm_prefix_envs)
        .unwrap_or_default();

    let (path_exists, path_is_valid_llvm) = if let Some(ref path) = configured_path {
        (path.exists(), is_valid_llvm(path))
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
        stale_llvm_prefix_envs,
        path_exists,
        path_is_valid_llvm,
        detected_path,
        config_matches_detected,
    }
}

/// Automatically configure LLVM for PECOS
///
/// This function determines the best LLVM installation to use and writes
/// it to `.cargo/config.toml` with `force=true`.
///
/// Priority order:
/// 1. `~/.pecos/deps/llvm-{version}` (PECOS-managed LLVM, new path)
/// 2. `~/.pecos/llvm` (legacy path)
/// 3. `LLVM_SYS_211_PREFIX` environment variable
/// 4. System LLVM (Homebrew, system paths, etc.)
///
/// # Errors
///
/// Returns an error if no suitable LLVM installation could be found
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
        pecos_llvm_paths.push(
            home_dir
                .join(".pecos")
                .join(format!("LLVM-{REQUIRED_VERSION}")),
        );
    }

    for pecos_llvm in pecos_llvm_paths {
        if is_valid_llvm(&pecos_llvm) {
            let project_root = project_root
                .or_else(get_repo_root_from_manifest)
                .or_else(find_cargo_project_root)
                .ok_or_else(|| Error::Config("Could not find Cargo project root".into()))?;

            write_cargo_config(&project_root, &pecos_llvm, true)?;
            return Ok(pecos_llvm);
        }
    }

    // Priority 2: Check LLVM_SYS_211_PREFIX
    if let Ok(sys_prefix) = std::env::var(LLVM_SYS_PREFIX_ENV) {
        let path = PathBuf::from(&sys_prefix);
        if is_valid_llvm(&path) {
            let project_root = project_root
                .or_else(get_repo_root_from_manifest)
                .or_else(find_cargo_project_root)
                .ok_or_else(|| Error::Config("Could not find Cargo project root".into()))?;

            write_cargo_config(&project_root, &path, true)?;
            return Ok(path);
        }
    }

    // Priority 3: Scan system for LLVM
    let repo_root = get_repo_root_from_manifest();
    if let Some(detected_path) = find_llvm(repo_root) {
        let project_root = project_root
            .or_else(get_repo_root_from_manifest)
            .or_else(find_cargo_project_root)
            .ok_or_else(|| Error::Config("Could not find Cargo project root".into()))?;

        write_cargo_config(&project_root, &detected_path, true)?;
        return Ok(detected_path);
    }

    Err(Error::Llvm(format!(
        "No suitable LLVM {REQUIRED_VERSION} installation found"
    )))
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
    // Forward slashes keep the value backslash-escape-free in TOML.
    let llvm_path_str = path_to_env_string(llvm_path);
    let mut cfg = crate::cargo_config::CargoConfig::open(project_root)?;
    cfg.remove_env_matching(is_stale_llvm_prefix_env)?;
    cfg.set_env(LLVM_SYS_PREFIX_ENV, &llvm_path_str, force)?;
    cfg.save()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_stale_llvm_prefix_envs() {
        let tmp = tempfile::tempdir().unwrap();
        let cargo_dir = tmp.path().join(".cargo");
        fs::create_dir_all(&cargo_dir).unwrap();
        fs::write(
            cargo_dir.join("config.toml"),
            r#"
[env]
LLVM_SYS_140_PREFIX = { value = "/old", force = true }
LLVM_SYS_211_PREFIX = { value = "/current", force = true }
LLVM_SYS_PREFIX = "/not-versioned"
FOO = "bar"
"#,
        )
        .unwrap();

        assert_eq!(
            read_stale_llvm_prefix_envs(tmp.path()),
            vec!["LLVM_SYS_140_PREFIX"]
        );
    }

    #[test]
    fn write_cargo_config_removes_stale_llvm_prefix_envs() {
        let tmp = tempfile::tempdir().unwrap();
        let cargo_dir = tmp.path().join(".cargo");
        fs::create_dir_all(&cargo_dir).unwrap();
        fs::write(
            cargo_dir.join("config.toml"),
            r#"
[env]
LLVM_SYS_140_PREFIX = { value = "/old", force = true }
FOO = "bar"
"#,
        )
        .unwrap();

        write_cargo_config(tmp.path(), Path::new("/llvm"), true).unwrap();

        let text = fs::read_to_string(cargo_dir.join("config.toml")).unwrap();
        let parsed: toml::Value = toml::from_str(&text).unwrap();
        assert!(parsed["env"].get("LLVM_SYS_140_PREFIX").is_none());
        assert_eq!(
            parsed["env"]["LLVM_SYS_211_PREFIX"]["value"],
            "/llvm".into()
        );
        assert_eq!(parsed["env"]["FOO"].as_str().unwrap(), "bar");
    }
}
