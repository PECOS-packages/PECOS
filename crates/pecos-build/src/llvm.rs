//! LLVM detection and management
//!
//! This module provides functionality to locate, install, and configure LLVM 21.1
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

/// LLVM version required by PECOS.
pub const REQUIRED_VERSION: &str = "21.1";

/// Cargo/llvm-sys environment variable for the required LLVM version.
pub const LLVM_SYS_PREFIX_ENV: &str = "LLVM_SYS_211_PREFIX";

/// Convert a path into a stable string for build environment variables and
/// Cargo config values.
///
/// Windows `canonicalize()` returns verbatim paths such as `\\?\C:\...`.
/// Most Rust and Windows APIs accept those, but bindgen's libclang loader does
/// not treat them as valid DLL search directories. Cargo config also does not
/// need the verbatim prefix, so strip it while keeping the path absolute.
#[must_use]
pub fn path_to_env_string(path: &Path) -> String {
    normalize_path_string(&path.to_string_lossy())
}

/// Normalize a stored path string before turning it back into a [`PathBuf`].
#[must_use]
pub fn normalize_path_string(path: &str) -> String {
    let path = path.replace('\\', "/");

    if let Some(rest) = path.strip_prefix("//?/UNC/") {
        return format!("//{rest}");
    }

    if let Some(rest) = path.strip_prefix("//?/")
        && is_windows_drive_path(rest)
    {
        return rest.to_string();
    }

    path
}

fn is_windows_drive_path(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' && bytes[2] == b'/'
}

/// Return whether an `llvm-config --version` string is compatible with PECOS.
#[must_use]
pub fn is_required_llvm_version(version: &str) -> bool {
    let version = version.trim();
    version == REQUIRED_VERSION
        || version
            .strip_prefix(REQUIRED_VERSION)
            .is_some_and(|rest| rest.starts_with('.'))
}

/// An LLVM tool resolved via a bare `PATH` lookup, with the version it reports.
///
/// Used to warn when a wrong-version tool (e.g. a system LLVM 14 `llvm-as`)
/// would be picked up at runtime instead of LLVM 21.1 — external tools such as
/// Selene resolve `llvm-as` straight from `PATH`, so a stale system LLVM ahead
/// of the PECOS one silently produces wrong results.
#[derive(Debug, Clone)]
pub struct PathToolReport {
    /// Tool name, e.g. `"llvm-as"`.
    pub tool: String,
    /// Location resolved from `PATH`.
    pub path: PathBuf,
    /// Version string reported by `<tool> --version`, if one could be parsed.
    pub version: Option<String>,
    /// Whether the reported version satisfies the required LLVM version.
    pub is_required: bool,
}

/// Resolve an executable via `PATH`, mirroring how a bare `Command::new(name)`
/// (and external tools like Selene) find it.
#[must_use]
pub fn which_on_path(exe_name: &str) -> Option<PathBuf> {
    let extensions: &[&str] = if cfg!(windows) { &[".exe"] } else { &[""] };
    crate::executable::which_in_path(exe_name, extensions)
}

/// Parse an LLVM `--version` output into a bare `X.Y.Z` version string.
///
/// Handles both `llvm-config --version` (prints `21.1.8`) and tools like
/// `llvm-as --version` (whose output contains `LLVM version 21.1.8`) by
/// returning the first dotted numeric token (at least `major.minor`).
#[must_use]
pub fn parse_llvm_version_output(output: &str) -> Option<String> {
    output
        .split_whitespace()
        .find(|token| {
            let mut parts = token.split('.');
            token.starts_with(|c: char| c.is_ascii_digit())
                && parts.clone().count() >= 2
                && parts.all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
        })
        .map(str::to_string)
}

/// Run `<tool_path> --version` and parse the LLVM version it reports.
#[must_use]
pub fn get_tool_llvm_version(tool_path: &Path) -> Option<String> {
    let output = Command::new(tool_path).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    parse_llvm_version_output(&String::from_utf8_lossy(&output.stdout))
}

/// Inspect the LLVM tools a bare `PATH` lookup resolves, reporting each one's
/// version and whether it is the required LLVM. A tool not found on `PATH` is
/// omitted from the result.
#[must_use]
pub fn inspect_path_llvm_tools(tools: &[&str]) -> Vec<PathToolReport> {
    tools
        .iter()
        .filter_map(|&tool| {
            let path = which_on_path(tool)?;
            let version = get_tool_llvm_version(&path);
            let is_required = version.as_deref().is_some_and(is_required_llvm_version);
            Some(PathToolReport {
                tool: tool.to_string(),
                path,
                version,
                is_required,
            })
        })
        .collect()
}

/// Find a compatible LLVM installation on the system.
///
/// This function searches for LLVM in the following priority order:
/// 1. PECOS deps directory: `~/.pecos/deps/llvm-{version}/`
/// 2. Legacy PECOS path: `~/.pecos/llvm/` (prints deprecation warning)
///    - Windows also checks: `~/.pecos/LLVM-{version}`
/// 3. Project-local installation (`llvm/` directory relative to repository root)
/// 4. System installations (platform-specific locations)
///
/// # Returns
/// - `Some(PathBuf)` if a compatible LLVM is found and valid
/// - `None` if a compatible LLVM is not found
#[must_use]
pub fn find_llvm(repo_root: Option<PathBuf>) -> Option<PathBuf> {
    // 1. Check versioned deps path: ~/.pecos/deps/llvm-{version}/
    if let Ok(deps_llvm) = crate::home::get_llvm_dir_path()
        && let Some(llvm_prefix) = valid_llvm_prefix(&deps_llvm)
    {
        return Some(llvm_prefix);
    }

    // 2. Check legacy top-level path: ~/.pecos/llvm/
    if let Some(home_dir) = dirs::home_dir() {
        let pecos_dir = home_dir.join(".pecos");

        #[cfg(target_os = "windows")]
        {
            let user_llvm_new = pecos_dir.join(format!("LLVM-{REQUIRED_VERSION}"));
            if let Some(llvm_prefix) = valid_llvm_prefix(&user_llvm_new) {
                crate::home::print_legacy_warning("LLVM", &llvm_prefix);
                return Some(llvm_prefix);
            }
        }

        let user_llvm_legacy = pecos_dir.join("llvm");
        if let Some(llvm_prefix) = valid_llvm_prefix(&user_llvm_legacy) {
            crate::home::print_legacy_warning("LLVM", &llvm_prefix);
            return Some(llvm_prefix);
        }
    }

    // 3. Check for project-local LLVM
    if let Some(root) = repo_root {
        let local_llvm = root.join("llvm");
        if let Some(llvm_prefix) = valid_llvm_prefix(&local_llvm) {
            return Some(llvm_prefix);
        }
    }

    // 4. Check system installations
    find_system_llvm()
}

fn valid_llvm_prefix(path: &Path) -> Option<PathBuf> {
    if is_valid_llvm(path) {
        return Some(path.to_path_buf());
    }

    #[cfg(target_os = "windows")]
    {
        let conda_library_prefix = path.join("Library");
        if is_valid_llvm(&conda_library_prefix) {
            return Some(conda_library_prefix);
        }
    }

    None
}

/// Find the LLVM installation Cargo should use for this project.
///
/// An explicit `.cargo/config.toml` setting takes priority because `cargo`
/// applies it to build scripts. If no valid project config exists, this falls
/// back to the normal managed/system detection order.
#[must_use]
pub fn find_configured_or_detected_llvm(repo_root: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(configured_path) = config::read_configured_llvm_path()
        && is_valid_llvm(&configured_path)
    {
        return Some(configured_path);
    }

    find_llvm(repo_root)
}

/// Find LLVM in system-wide locations (platform-specific)
fn find_system_llvm() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = Command::new("brew").args(["--prefix", "llvm@21"]).output()
            && output.status.success()
        {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let path = PathBuf::from(path_str);
            if is_valid_llvm(&path) {
                return Some(path);
            }
        }

        for path_str in ["/opt/homebrew/opt/llvm@21", "/usr/local/opt/llvm@21"] {
            let llvm_path = PathBuf::from(path_str);
            if is_valid_llvm(&llvm_path) {
                return Some(llvm_path);
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(output) = Command::new("llvm-config-21").arg("--prefix").output()
            && output.status.success()
        {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let path = PathBuf::from(path_str);
            if is_valid_llvm(&path) {
                return Some(path);
            }
        }

        for path_str in [
            "/usr/lib/llvm-21",
            "/usr/local/llvm-21",
            "/usr/lib/x86_64-linux-gnu/llvm-21",
        ] {
            let llvm_path = PathBuf::from(path_str);
            if is_valid_llvm(&llvm_path) {
                return Some(llvm_path);
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        for path_str in [
            "C:\\Program Files\\LLVM",
            "C:\\LLVM",
            "C:\\Program Files\\LLVM-21",
            "C:\\LLVM-21",
        ] {
            let llvm_path = PathBuf::from(path_str);
            if is_valid_llvm(&llvm_path) {
                return Some(llvm_path);
            }
        }
    }

    None
}

/// Check if a given path contains a compatible LLVM installation
#[must_use]
pub fn is_valid_llvm(path: &Path) -> bool {
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
        return is_required_llvm_version(&version);
    }

    false
}

/// Get the version of LLVM at the given path
///
/// # Errors
///
/// Returns an error if LLVM is not found or version cannot be determined
pub fn get_llvm_version(path: &Path) -> Result<String> {
    let output = Command::new(llvm_config_path(path))
        .arg("--version")
        .output()
        .map_err(|e| Error::Llvm(format!("Failed to run llvm-config: {e}")))?;

    if !output.status.success() {
        return Err(Error::Llvm("llvm-config returned non-zero status".into()));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Get LLVM's configured shared/static library mode.
///
/// # Errors
///
/// Returns an error if `llvm-config --shared-mode` fails.
pub fn get_llvm_shared_mode(path: &Path) -> Result<String> {
    let output = Command::new(llvm_config_path(path))
        .arg("--shared-mode")
        .output()
        .map_err(|e| Error::Llvm(format!("Failed to run llvm-config: {e}")))?;

    if !output.status.success() {
        return Err(Error::Llvm(
            "llvm-config --shared-mode returned non-zero status".into(),
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Get the shared LLVM library names reported by `llvm-config`, if available.
#[must_use]
pub fn get_llvm_shared_libraries(path: &Path) -> Option<String> {
    let output = Command::new(llvm_config_path(path))
        .args(["--libnames", "--link-shared", "core"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let libraries = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if libraries.is_empty() {
        None
    } else {
        Some(libraries)
    }
}

/// Get LLVM's library directory as reported by `llvm-config --libdir`.
///
/// # Errors
///
/// Returns an error if `llvm-config --libdir` fails.
pub fn get_llvm_libdir(path: &Path) -> Result<PathBuf> {
    let output = Command::new(llvm_config_path(path))
        .arg("--libdir")
        .output()
        .map_err(|e| Error::Llvm(format!("Failed to run llvm-config: {e}")))?;

    if !output.status.success() {
        return Err(Error::Llvm(
            "llvm-config --libdir returned non-zero status".into(),
        ));
    }

    Ok(PathBuf::from(
        String::from_utf8_lossy(&output.stdout).trim(),
    ))
}

fn llvm_config_path(path: &Path) -> PathBuf {
    let exe_name = if cfg!(windows) {
        "llvm-config.exe"
    } else {
        "llvm-config"
    };

    path.join("bin").join(exe_name)
}

/// Find a specific LLVM tool by name
#[must_use]
pub fn find_tool(tool_name: &str) -> Option<PathBuf> {
    let repo_root = get_repo_root_from_manifest();
    let llvm_path = find_configured_or_detected_llvm(repo_root)?;

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

/// Print a helpful error message when the required LLVM version is not found
pub fn print_llvm_not_found_error() {
    let cmd = get_pecos_command();

    eprintln!("\n═══════════════════════════════════════════════════════════════");
    eprintln!("ERROR: LLVM {REQUIRED_VERSION} not found!");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();
    eprintln!("PECOS requires LLVM version {REQUIRED_VERSION} for QIS program execution.");
    eprintln!();
    if installer::managed_install_unavailable_reason().is_none() {
        eprintln!("Option 1 - Install LLVM {REQUIRED_VERSION} for PECOS (recommended):");
        eprintln!();
        eprintln!("    {cmd} install llvm");
        eprintln!();
    }

    #[cfg(target_os = "macos")]
    {
        eprintln!("Use system LLVM via Homebrew:");
        eprintln!();
        eprintln!("    brew install llvm@21");
        eprintln!("    {cmd} llvm configure");
        eprintln!();
    }

    #[cfg(target_os = "linux")]
    {
        eprintln!("Option 2 - Use system LLVM via package manager:");
        eprintln!();
        eprintln!("    Install LLVM 21.1 through your distribution packages if available");
        eprintln!("    {cmd} llvm configure");
        eprintln!();
    }

    #[cfg(target_os = "windows")]
    {
        eprintln!("The official Windows LLVM installer is not sufficient for PECOS.");
        eprintln!("Use scripts\\ci\\install-llvm-21-windows.ps1 for the conda-forge");
        eprintln!("LLVM 21.1 toolchain, then configure its Library prefix:");
        eprintln!();
        eprintln!("    {cmd} llvm configure %USERPROFILE%\\.pecos\\deps\\llvm-21.1\\Library");
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

    #[test]
    fn required_llvm_version_matches_only_21_1_series() {
        assert!(is_required_llvm_version("21.1"));
        assert!(is_required_llvm_version("21.1.8"));
        assert!(is_required_llvm_version("21.1.8git"));
        assert!(!is_required_llvm_version("21.0.9"));
        assert!(!is_required_llvm_version("21.10.0"));
        assert!(!is_required_llvm_version("22.0.0"));
    }

    #[test]
    fn parse_llvm_version_handles_config_and_tool_output() {
        // `llvm-config --version` prints the bare version.
        assert_eq!(
            parse_llvm_version_output("21.1.8\n").as_deref(),
            Some("21.1.8")
        );
        // `llvm-as --version` prints a block; the first dotted numeric token is
        // the LLVM version, not the target triple or CPU line.
        let llvm_as = "  LLVM (http://llvm.org/):\n    LLVM version 14.0.0\n    \
                       Optimized build.\n    Default target: x86_64-pc-linux-gnu\n";
        assert_eq!(
            parse_llvm_version_output(llvm_as).as_deref(),
            Some("14.0.0")
        );
        assert_eq!(parse_llvm_version_output("no version here"), None);
    }

    #[test]
    fn path_tool_report_flags_wrong_version() {
        // Version parsing + is_required together decide whether a PATH tool is
        // acceptable; a system LLVM 14 must be flagged, 21.1.x accepted.
        assert!(!is_required_llvm_version(
            &parse_llvm_version_output("LLVM version 14.0.0").unwrap()
        ));
        assert!(is_required_llvm_version(
            &parse_llvm_version_output("LLVM version 21.1.8").unwrap()
        ));
    }

    #[test]
    fn normalize_path_string_strips_windows_verbatim_drive_prefix() {
        assert_eq!(
            normalize_path_string(r"\\?\C:\Users\runneradmin\.pecos\deps\llvm-21.1\Library"),
            "C:/Users/runneradmin/.pecos/deps/llvm-21.1/Library"
        );
        assert_eq!(
            normalize_path_string("//?/C:/Users/runneradmin/.pecos/deps/llvm-21.1/Library"),
            "C:/Users/runneradmin/.pecos/deps/llvm-21.1/Library"
        );
    }

    #[test]
    fn normalize_path_string_strips_windows_verbatim_unc_prefix() {
        assert_eq!(
            normalize_path_string(r"\\?\UNC\server\share\llvm-21.1"),
            "//server/share/llvm-21.1"
        );
    }

    #[test]
    fn normalize_path_string_leaves_non_verbatim_paths_alone() {
        assert_eq!(
            normalize_path_string("/home/ciaranra/.pecos/deps/llvm-21.1"),
            "/home/ciaranra/.pecos/deps/llvm-21.1"
        );
        assert_eq!(
            normalize_path_string("//server/share/llvm-21.1"),
            "//server/share/llvm-21.1"
        );
    }
}
