//! LLVM detection and management
//!
//! This module provides functionality to locate, install, and configure LLVM 14
//! for PECOS across different platforms.

pub mod config;
pub mod installer;

use crate::errors::{Error, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Determine the best command prefix for running pecos CLI commands.
///
/// Returns the appropriate command prefix based on what's available:
/// - `"pecos"` if the pecos CLI is installed
/// - `"cargo run -p pecos --"` as fallback
#[must_use]
pub fn get_pecos_command() -> &'static str {
    // Check if pecos is in PATH
    if Command::new("pecos")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
    {
        return "pecos";
    }

    // Fall back to cargo run
    "cargo run -p pecos --"
}

/// LLVM version required by PECOS
pub const REQUIRED_VERSION: &str = "14";

/// Find LLVM 14 installation on the system.
///
/// This function searches for LLVM 14 in the following priority order:
/// 1. PECOS deps directory: `~/.pecos/deps/llvm/`
/// 2. Legacy PECOS path: `~/.pecos/llvm/` (prints deprecation warning)
///    - Windows also checks: `~/.pecos/LLVM-14`
/// 3. Project-local installation (`llvm/` directory relative to repository root)
/// 4. System installations (platform-specific locations)
///
/// # Returns
/// - `Some(PathBuf)` if LLVM 14 is found and valid
/// - `None` if LLVM 14 is not found
#[must_use]
pub fn find_llvm_14(repo_root: Option<PathBuf>) -> Option<PathBuf> {
    // 1. Check new deps path: ~/.pecos/deps/llvm/
    if let Ok(deps_llvm) = crate::home::get_llvm_dir_path()
        && is_valid_llvm_14(&deps_llvm)
    {
        return Some(deps_llvm);
    }

    // 2. Check legacy top-level path: ~/.pecos/llvm/
    if let Some(home_dir) = dirs::home_dir() {
        let pecos_dir = home_dir.join(".pecos");

        #[cfg(target_os = "windows")]
        {
            let user_llvm_new = pecos_dir.join("LLVM-14");
            if is_valid_llvm_14(&user_llvm_new) {
                crate::home::print_legacy_warning("LLVM", &user_llvm_new);
                return Some(user_llvm_new);
            }
        }

        let user_llvm_legacy = pecos_dir.join("llvm");
        if is_valid_llvm_14(&user_llvm_legacy) {
            crate::home::print_legacy_warning("LLVM", &user_llvm_legacy);
            return Some(user_llvm_legacy);
        }
    }

    // 3. Check for project-local LLVM
    if let Some(root) = repo_root {
        let local_llvm = root.join("llvm");
        if is_valid_llvm_14(&local_llvm) {
            return Some(local_llvm);
        }
    }

    // 4. Check system installations
    find_system_llvm_14()
}

/// Find LLVM 14 in system-wide locations (platform-specific)
fn find_system_llvm_14() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = Command::new("brew").args(["--prefix", "llvm@14"]).output()
            && output.status.success()
        {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let path = PathBuf::from(path_str);
            if is_valid_llvm_14(&path) {
                return Some(path);
            }
        }

        for path_str in ["/opt/homebrew/opt/llvm@14", "/usr/local/opt/llvm@14"] {
            let llvm_path = PathBuf::from(path_str);
            if is_valid_llvm_14(&llvm_path) {
                return Some(llvm_path);
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(output) = Command::new("llvm-config-14").arg("--prefix").output()
            && output.status.success()
        {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let path = PathBuf::from(path_str);
            if is_valid_llvm_14(&path) {
                return Some(path);
            }
        }

        for path_str in [
            "/usr/lib/llvm-14",
            "/usr/local/llvm-14",
            "/usr/lib/x86_64-linux-gnu/llvm-14",
        ] {
            let llvm_path = PathBuf::from(path_str);
            if is_valid_llvm_14(&llvm_path) {
                return Some(llvm_path);
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        for path_str in [
            "C:\\Program Files\\LLVM",
            "C:\\LLVM",
            "C:\\Program Files\\LLVM-14",
            "C:\\LLVM-14",
        ] {
            let llvm_path = PathBuf::from(path_str);
            if is_valid_llvm_14(&llvm_path) {
                return Some(llvm_path);
            }
        }
    }

    None
}

/// Check if a given path contains a valid LLVM 14 installation
#[must_use]
pub fn is_valid_llvm_14(path: &Path) -> bool {
    if !path.exists() {
        return false;
    }

    #[cfg(target_os = "windows")]
    let llvm_config = path.join("bin").join("llvm-config.exe");

    #[cfg(not(target_os = "windows"))]
    let llvm_config = path.join("bin").join("llvm-config");

    if !llvm_config.exists() {
        return false;
    }

    if let Ok(output) = Command::new(&llvm_config).arg("--version").output()
        && output.status.success()
    {
        let version = String::from_utf8_lossy(&output.stdout);
        return version.starts_with("14.");
    }

    false
}

/// Get the version of LLVM at the given path
///
/// # Errors
///
/// Returns an error if LLVM is not found or version cannot be determined
pub fn get_llvm_version(path: &Path) -> Result<String> {
    #[cfg(target_os = "windows")]
    let llvm_config = path.join("bin").join("llvm-config.exe");

    #[cfg(not(target_os = "windows"))]
    let llvm_config = path.join("bin").join("llvm-config");

    let output = Command::new(&llvm_config)
        .arg("--version")
        .output()
        .map_err(|e| Error::Llvm(format!("Failed to run llvm-config: {e}")))?;

    if !output.status.success() {
        return Err(Error::Llvm("llvm-config returned non-zero status".into()));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Find a specific LLVM tool by name
#[must_use]
pub fn find_tool(tool_name: &str) -> Option<PathBuf> {
    let repo_root = get_repo_root_from_manifest();
    let llvm_path = find_llvm_14(repo_root)?;

    let tool_path = if cfg!(windows) {
        llvm_path.join("bin").join(format!("{tool_name}.exe"))
    } else {
        llvm_path.join("bin").join(tool_name)
    };

    if tool_path.exists() {
        Some(tool_path)
    } else {
        None
    }
}

/// Get the repository root from `CARGO_MANIFEST_DIR`
#[must_use]
pub fn get_repo_root_from_manifest() -> Option<PathBuf> {
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let mut path = PathBuf::from(manifest_dir);
        if path.pop() && path.pop() {
            return Some(path);
        }
    }
    None
}

/// Find the Cargo project root by searching for Cargo.toml.
///
/// Prefers a workspace root: walks all the way up from cwd and returns the
/// first ancestor whose `Cargo.toml` contains a `[workspace]` section.
/// Falls back to the nearest `Cargo.toml` or `Cargo.lock` (same behavior as
/// before for non-workspace projects).
#[must_use]
pub fn find_cargo_project_root() -> Option<PathBuf> {
    let current_dir = std::env::current_dir().ok()?;
    find_cargo_project_root_from(&current_dir)
}

/// Core logic for [`find_cargo_project_root`], starting from the given path.
fn find_cargo_project_root_from(start: &Path) -> Option<PathBuf> {
    let mut path = start;
    let mut first_match: Option<PathBuf> = None;

    loop {
        let cargo_toml = path.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(contents) = std::fs::read_to_string(&cargo_toml)
                && contents.contains("[workspace]")
            {
                return Some(path.to_path_buf());
            }
            if first_match.is_none() {
                first_match = Some(path.to_path_buf());
            }
        }
        if first_match.is_none() && path.join("Cargo.lock").exists() {
            first_match = Some(path.to_path_buf());
        }
        match path.parent() {
            Some(parent) => path = parent,
            None => break,
        }
    }

    first_match
}

/// Print a helpful error message when LLVM 14 is not found
pub fn print_llvm_not_found_error() {
    let cmd = get_pecos_command();

    eprintln!("\n═══════════════════════════════════════════════════════════════");
    eprintln!("ERROR: LLVM 14 not found!");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();
    eprintln!("PECOS requires LLVM version 14 for QIS program execution.");
    eprintln!();
    eprintln!("Option 1 - Install LLVM 14 for PECOS (recommended):");
    eprintln!();
    eprintln!("    {cmd} install llvm");
    eprintln!();

    #[cfg(target_os = "macos")]
    {
        eprintln!("Option 2 - Use system LLVM via Homebrew:");
        eprintln!();
        eprintln!("    brew install llvm@14");
        eprintln!("    {cmd} llvm configure");
        eprintln!();
    }

    #[cfg(target_os = "linux")]
    {
        eprintln!("Option 2 - Use system LLVM via package manager:");
        eprintln!();
        eprintln!("    sudo apt install llvm-14  # Debian/Ubuntu");
        eprintln!("    {cmd} llvm configure");
        eprintln!();
    }

    #[cfg(target_os = "windows")]
    {
        eprintln!("For Windows, use the PECOS installer (Option 1) above.");
        eprintln!();
    }

    eprintln!("═══════════════════════════════════════════════════════════════\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn workspace_root_preferred_over_subcrate() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // workspace root
        fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/foo\"]\n",
        )
        .unwrap();

        // subcrate
        let subcrate = root.join("crates").join("foo");
        fs::create_dir_all(&subcrate).unwrap();
        fs::write(
            subcrate.join("Cargo.toml"),
            "[package]\nname = \"foo\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();

        // Starting from the subcrate should return the workspace root.
        let result = find_cargo_project_root_from(&subcrate);
        assert_eq!(result.as_deref(), Some(root));
    }

    #[test]
    fn returns_first_cargo_toml_when_no_workspace() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // standalone project (no [workspace] section)
        let project = root.join("project");
        fs::create_dir_all(&project).unwrap();
        fs::write(
            project.join("Cargo.toml"),
            "[package]\nname = \"standalone\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();

        let subdir = project.join("src");
        fs::create_dir_all(&subdir).unwrap();

        let result = find_cargo_project_root_from(&subdir);
        assert_eq!(result.as_deref(), Some(project.as_path()));
    }

    #[test]
    fn returns_cargo_lock_dir_when_no_cargo_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // Directory with only Cargo.lock
        fs::write(root.join("Cargo.lock"), "").unwrap();

        let subdir = root.join("deep").join("nested");
        fs::create_dir_all(&subdir).unwrap();

        let result = find_cargo_project_root_from(&subdir);
        assert_eq!(result.as_deref(), Some(root));
    }

    #[test]
    fn returns_none_when_no_cargo_files() {
        let tmp = tempfile::tempdir().unwrap();
        let empty = tmp.path().join("empty");
        fs::create_dir_all(&empty).unwrap();

        let result = find_cargo_project_root_from(&empty);
        assert_eq!(result, None);
    }

    #[test]
    fn workspace_root_found_above_intermediate_crate() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // workspace root
        fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/*\"]\n",
        )
        .unwrap();

        // intermediate crate (not a workspace)
        let mid = root.join("crates").join("mid");
        fs::create_dir_all(&mid).unwrap();
        fs::write(
            mid.join("Cargo.toml"),
            "[package]\nname = \"mid\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();

        // deep src directory
        let deep = mid.join("src").join("submod");
        fs::create_dir_all(&deep).unwrap();

        let result = find_cargo_project_root_from(&deep);
        assert_eq!(result.as_deref(), Some(root));
    }
}
