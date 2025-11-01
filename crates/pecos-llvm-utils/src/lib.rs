//! LLVM detection and utilities for PECOS
//!
//! This crate provides functionality to locate and install LLVM 14 across different platforms.
//! It's primarily used by build scripts but can also be used standalone via the `pecos-llvm` binary.

pub mod installer;

use std::path::{Path, PathBuf};
use std::process::Command;

/// Find LLVM 14 installation on the system.
///
/// This function searches for LLVM 14 in the following priority order:
/// 1. Home directory:
///    - Windows: ~/.pecos/LLVM-14 (new) or ~/.pecos/llvm (legacy)
///    - Unix: ~/.pecos/llvm
/// 2. Project-local installation (llvm/ directory relative to repository root)
/// 3. System installations (platform-specific locations)
///
/// # Returns
/// - `Some(PathBuf)` if LLVM 14 is found and valid
/// - `None` if LLVM 14 is not found
///
/// # Example
/// ```no_run
/// use pecos_llvm_utils::find_llvm_14;
///
/// if let Some(llvm_path) = find_llvm_14(None) {
///     println!("Found LLVM 14 at: {}", llvm_path.display());
/// } else {
///     eprintln!("LLVM 14 not found!");
/// }
/// ```
#[must_use]
pub fn find_llvm_14(repo_root: Option<PathBuf>) -> Option<PathBuf> {
    // 1. Check home directory
    if let Some(home_dir) = dirs::home_dir() {
        let pecos_dir = home_dir.join(".pecos");

        // On Windows, check new location first (LLVM-14), then legacy (llvm)
        #[cfg(target_os = "windows")]
        {
            let user_llvm_new = pecos_dir.join("LLVM-14");
            if is_valid_llvm_14(&user_llvm_new) {
                return Some(user_llvm_new);
            }
            let user_llvm_legacy = pecos_dir.join("llvm");
            if is_valid_llvm_14(&user_llvm_legacy) {
                return Some(user_llvm_legacy);
            }
        }

        // On Unix, check standard location
        #[cfg(not(target_os = "windows"))]
        {
            let user_llvm = pecos_dir.join("llvm");
            if is_valid_llvm_14(&user_llvm) {
                return Some(user_llvm);
            }
        }
    }

    // 2. Check for project-local LLVM (for backward compatibility)
    if let Some(root) = repo_root {
        let local_llvm = root.join("llvm");
        if is_valid_llvm_14(&local_llvm) {
            return Some(local_llvm);
        }
    }

    // 3. Check system installations
    find_system_llvm_14()
}

/// Find LLVM 14 in system-wide locations (platform-specific)
fn find_system_llvm_14() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        // Try Homebrew installation via brew command
        if let Ok(output) = Command::new("brew").args(["--prefix", "llvm@14"]).output()
            && output.status.success()
        {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let path = PathBuf::from(path_str);
            if is_valid_llvm_14(&path) {
                return Some(path);
            }
        }

        // Try common Homebrew paths (in case brew command isn't available)
        for path_str in [
            "/opt/homebrew/opt/llvm@14", // Apple Silicon
            "/usr/local/opt/llvm@14",    // Intel Mac
        ] {
            let llvm_path = PathBuf::from(path_str);
            if is_valid_llvm_14(&llvm_path) {
                return Some(llvm_path);
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        // Check if llvm-config-14 is in PATH and get its prefix
        if let Ok(output) = Command::new("llvm-config-14").arg("--prefix").output()
            && output.status.success()
        {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let path = PathBuf::from(path_str);
            if is_valid_llvm_14(&path) {
                return Some(path);
            }
        }

        // Try common Linux installation paths
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
        // Try common Windows installation paths
        // Note: The official LLVM Windows installer (LLVM-*.exe) is toolchain-only
        // and lacks llvm-config.exe and development headers.
        // Users need a full development package (e.g., from community sources).
        for path_str in [
            "C:\\Program Files\\LLVM",    // Official installer (usually incomplete)
            "C:\\LLVM",                   // Custom installation
            "C:\\Program Files\\LLVM-14", // Versioned installation
            "C:\\LLVM-14",                // Versioned custom installation
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
///
/// # Arguments
/// * `path` - Path to check for LLVM installation
///
/// # Returns
/// `true` if the path contains a valid LLVM 14 installation, `false` otherwise
#[must_use]
pub fn is_valid_llvm_14(path: &Path) -> bool {
    // Check if the path exists
    if !path.exists() {
        return false;
    }

    // Determine llvm-config path based on platform
    #[cfg(target_os = "windows")]
    let llvm_config = path.join("bin").join("llvm-config.exe");

    #[cfg(not(target_os = "windows"))]
    let llvm_config = path.join("bin").join("llvm-config");

    if !llvm_config.exists() {
        return false;
    }

    // Verify it's LLVM 14 by checking the version
    if let Ok(output) = Command::new(&llvm_config).arg("--version").output()
        && output.status.success()
    {
        let version = String::from_utf8_lossy(&output.stdout);
        return version.starts_with("14.");
    }

    false
}

/// Print a helpful error message when LLVM 14 is not found
pub fn print_llvm_not_found_error() {
    eprintln!("\n═══════════════════════════════════════════════════════════════");
    eprintln!("ERROR: LLVM 14 not found!");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();
    eprintln!("PECOS requires LLVM version 14 for LLVM IR/QIR execution features.");
    eprintln!();
    eprintln!("To install LLVM 14:");
    eprintln!();
    eprintln!("  Automated installation (all platforms):");
    eprintln!("    cargo run -p pecos-llvm-utils --bin pecos-llvm --release -- install");
    eprintln!();

    #[cfg(target_os = "macos")]
    {
        eprintln!("  Or install via Homebrew:");
        eprintln!("    brew install llvm@14");
        eprintln!();
        eprintln!("  Then the build system will auto-detect it, or set:");
        eprintln!("    export LLVM_SYS_140_PREFIX=$(brew --prefix llvm@14)");
    }

    #[cfg(target_os = "linux")]
    {
        eprintln!("  Or install via package manager:");
        eprintln!("    sudo apt install llvm-14  # Debian/Ubuntu");
        eprintln!();
        eprintln!("  The build system will auto-detect most installations, or set:");
        eprintln!("    export LLVM_SYS_140_PREFIX=/usr/lib/llvm-14");
    }

    #[cfg(target_os = "windows")]
    {
        eprintln!("  For system-wide installation on Windows:");
        eprintln!("    IMPORTANT: The official LLVM Windows installer (LLVM-*.exe) is");
        eprintln!("    toolchain-only and lacks development files (llvm-config, headers).");
        eprintln!();
        eprintln!("    You need a FULL DEVELOPMENT package from:");
        eprintln!("    - https://github.com/bitgate/llvm-windows-full-builds (recommended)");
        eprintln!("    - https://github.com/vovkos/llvm-package-windows");
        eprintln!("    - Build from source: https://llvm.org/docs/GettingStarted.html");
        eprintln!();
        eprintln!("  After installation, set:");
        eprintln!("    set LLVM_SYS_140_PREFIX=C:\\path\\to\\llvm");
    }

    eprintln!();
    eprintln!("Alternatively, you can build without LLVM support:");
    eprintln!("  cargo build --no-default-features");
    eprintln!();
    eprintln!("For more details, see:");
    eprintln!("  https://quantum-pecos.readthedocs.io/");
    eprintln!("═══════════════════════════════════════════════════════════════\n");
}

/// Automatically configure LLVM for PECOS
///
/// This function determines the best LLVM 14 installation to use and writes
/// it to `.cargo/config.toml` with force=true. This is the authoritative
/// configuration function for PECOS.
///
/// Priority order:
/// 1. ~/.pecos/llvm (or LLVM-14 on Windows) - PECOS-managed LLVM (if it exists)
/// 2. `LLVM_SYS_140_PREFIX` environment variable (if set and valid)
/// 3. System LLVM 14 (Homebrew, system paths, etc.)
///
/// # Arguments
/// * `project_root` - Optional path to the Cargo project root. If None, attempts to find it.
///
/// # Errors
/// Returns an error if:
/// - No suitable LLVM 14 installation could be found
/// - The Cargo project root could not be determined
/// - Writing to `.cargo/config.toml` fails
///
/// # Returns
/// * `Ok(PathBuf)` - The path that was configured
pub fn auto_configure_llvm(
    project_root: Option<PathBuf>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    use std::env;

    // Priority 1: Check ~/.pecos/ for PECOS-managed LLVM
    // Uses find_llvm_14 which checks platform-appropriate paths
    if let Some(home_dir) = dirs::home_dir() {
        let pecos_dir = home_dir.join(".pecos");

        // Windows: checks LLVM-14 (custom) then llvm (standard)
        // Unix: checks llvm only
        #[cfg(target_os = "windows")]
        let pecos_llvm_paths = vec![
            pecos_dir.join("LLVM-14"), // Custom Windows naming
            pecos_dir.join("llvm"),    // Standard naming
        ];

        #[cfg(not(target_os = "windows"))]
        let pecos_llvm_paths = vec![pecos_dir.join("llvm")];

        for pecos_llvm in pecos_llvm_paths {
            if is_valid_llvm_14(&pecos_llvm) {
                // Found PECOS-managed LLVM, configure it
                let project_root = project_root
                    .or_else(get_repo_root_from_manifest)
                    .or_else(find_cargo_project_root)
                    .ok_or("Could not find Cargo project root")?;

                write_cargo_config(&project_root, &pecos_llvm, true)?;
                return Ok(pecos_llvm);
            }
        }
    }

    // Priority 2: Check shell LLVM_SYS_140_PREFIX
    if let Ok(sys_prefix) = env::var("LLVM_SYS_140_PREFIX") {
        let path = PathBuf::from(&sys_prefix);
        if is_valid_llvm_14(&path) {
            // Shell env var points to valid LLVM, configure it
            let project_root = project_root
                .or_else(get_repo_root_from_manifest)
                .or_else(find_cargo_project_root)
                .ok_or("Could not find Cargo project root")?;

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
            .ok_or("Could not find Cargo project root")?;

        write_cargo_config(&project_root, &detected_path, true)?;
        return Ok(detected_path);
    }

    // No LLVM 14 found anywhere
    Err("No suitable LLVM 14 installation found".into())
}

/// Get the repository root from `CARGO_MANIFEST_DIR`
///
/// This assumes the crate is located at `crates/<crate-name>` in the repository
#[must_use]
pub fn get_repo_root_from_manifest() -> Option<PathBuf> {
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let mut path = PathBuf::from(manifest_dir);
        // Go up from crates/<crate-name> to repository root
        if path.pop() && path.pop() {
            return Some(path);
        }
    }
    None
}

/// Find the Cargo project root by searching for Cargo.toml
///
/// Starts from the current directory and walks up the directory tree
/// until it finds a directory containing Cargo.toml or Cargo.lock
#[must_use]
pub fn find_cargo_project_root() -> Option<PathBuf> {
    let current_dir = std::env::current_dir().ok()?;
    let mut path = current_dir.as_path();

    loop {
        if path.join("Cargo.toml").exists() || path.join("Cargo.lock").exists() {
            return Some(path.to_path_buf());
        }

        path = path.parent()?;
    }
}

/// Find a specific LLVM tool by name
///
/// This function locates a specific LLVM tool (e.g., `llvm-as`, `clang`) by:
/// 1. Finding the LLVM 14 installation
/// 2. Constructing the tool path with proper OS-specific naming (e.g., `.exe` on Windows)
/// 3. Verifying the tool exists
///
/// # Arguments
/// * `tool_name` - The name of the tool (e.g., "llvm-as", "clang", "llvm-link")
///
/// # Returns
/// * `Some(PathBuf)` if the tool is found
/// * `None` if LLVM 14 is not found or the tool doesn't exist
///
/// # Example
/// ```no_run
/// use pecos_llvm_utils::find_tool;
///
/// if let Some(llvm_as) = find_tool("llvm-as") {
///     println!("Found llvm-as at: {}", llvm_as.display());
/// }
/// ```
#[must_use]
pub fn find_tool(tool_name: &str) -> Option<PathBuf> {
    // Find LLVM installation
    let repo_root = get_repo_root_from_manifest();
    let llvm_path = find_llvm_14(repo_root)?;

    // Construct tool path with OS-specific extension
    let tool_path = if cfg!(windows) {
        llvm_path.join("bin").join(format!("{tool_name}.exe"))
    } else {
        llvm_path.join("bin").join(tool_name)
    };

    // Verify the tool exists
    if tool_path.exists() {
        Some(tool_path)
    } else {
        None
    }
}

/// Write or update .cargo/config.toml with LLVM configuration
///
/// # Arguments
/// * `project_root` - Path to the Cargo project root
/// * `llvm_path` - Path to the LLVM installation
/// * `force` - If true, use force=true to override shell environment variables
///
/// # Errors
/// Returns an error if:
/// - Creating the `.cargo` directory fails
/// - Reading or writing to `.cargo/config.toml` fails
///
/// # Returns
/// `Ok(())` if successful
pub fn write_cargo_config(
    project_root: &Path,
    llvm_path: &Path,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;

    let cargo_dir = project_root.join(".cargo");
    let config_path = cargo_dir.join("config.toml");

    // Create .cargo directory if it doesn't exist
    fs::create_dir_all(&cargo_dir)?;

    // Convert path to forward slashes for TOML compatibility (Windows accepts forward slashes)
    let llvm_path_str = llvm_path.to_string_lossy().replace('\\', "/");

    // Format the LLVM_SYS_140_PREFIX line based on force flag
    let llvm_line = if force {
        format!("LLVM_SYS_140_PREFIX = {{ value = \"{llvm_path_str}\", force = true }}")
    } else {
        format!("LLVM_SYS_140_PREFIX = \"{llvm_path_str}\"")
    };

    // Read existing config or start with empty string
    let existing_content = fs::read_to_string(&config_path).unwrap_or_default();

    // Check if config already has LLVM_SYS_140_PREFIX
    if existing_content.contains("LLVM_SYS_140_PREFIX") {
        // Check if it's set to the same value (either simple or force format)
        let simple_format = format!("LLVM_SYS_140_PREFIX = \"{llvm_path_str}\"");
        let force_format =
            format!("LLVM_SYS_140_PREFIX = {{ value = \"{llvm_path_str}\", force = true }}");

        if existing_content.contains(&simple_format) || existing_content.contains(&force_format) {
            // Already configured correctly (might be different format, but same path)
            // If force flag changed, we should still update
            if (force && existing_content.contains(&force_format))
                || (!force && existing_content.contains(&simple_format))
            {
                return Ok(());
            }
        }

        // Configuration exists but needs updating - replace it
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

            // Track if we're in the [env] section
            if trimmed.starts_with('[') {
                in_env_section = trimmed == "[env]";
            }

            // Update LLVM_SYS_140_PREFIX if we find it
            if in_env_section && trimmed.starts_with("LLVM_SYS_140_PREFIX") {
                new_lines.push(llvm_line.clone());
                updated = true;

                // If old format was multi-line (with braces), skip continuation lines
                if trimmed.contains('{') && !trimmed.contains('}') {
                    // Count lines until we find closing brace
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_llvm_14() {
        // This test will only pass if LLVM 14 is installed on the system
        // Skip it in CI if LLVM is not available
        if let Some(path) = find_llvm_14(None) {
            println!("Found LLVM 14 at: {}", path.display());
            assert!(is_valid_llvm_14(&path));
        } else {
            println!("LLVM 14 not found (this is okay for CI)");
        }
    }
}
