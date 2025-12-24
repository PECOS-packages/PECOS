//! Utility functions for Selene runtime plugins
//!
//! This module provides convenient access to Selene runtime implementations.
//! The runtimes are automatically built when you build this crate if the
//! Selene repository is found at ../selene (relative to PECOS).

use crate::SeleneRuntime;
use std::path::PathBuf;

/// Error type for runtime fetching
#[derive(Debug)]
pub enum RuntimeFetchError {
    IoError(std::io::Error),
    DownloadError(String),
    InvalidPath(String),
}

impl std::fmt::Display for RuntimeFetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "IO error: {e}"),
            Self::DownloadError(msg) => write!(f, "Download error: {msg}"),
            Self::InvalidPath(msg) => write!(f, "Invalid path: {msg}"),
        }
    }
}

impl std::error::Error for RuntimeFetchError {}

impl From<std::io::Error> for RuntimeFetchError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

/// Create a Selene Simple Runtime
///
/// This loads the Selene Simple runtime plugin that was built by the build script.
/// The runtime is expected to be at `../selene/target/release/libselene_simple_runtime.so`
/// (relative to the PECOS workspace).
///
/// # Example
/// ```rust
/// use pecos_qis_selene::{selene_simple_runtime};
/// use pecos_qis_core::{qis_engine, QisEngine};
/// use pecos_engines::ClassicalControlEngineBuilder;
/// use pecos_qis_ffi_types::OperationCollector;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Load the simple runtime (built during compilation)
/// match selene_simple_runtime() {
///     Ok(runtime) => {
///         let interface = OperationCollector::new();
///         let engine = qis_engine().runtime(runtime).program(interface).build()?;
///         // Engine is ready to use
///     }
///     Err(e) => {
///         // Runtime not built - Selene repository not found
///         eprintln!("Simple runtime not available: {}", e);
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Errors
/// Returns an error if the Selene simple runtime library cannot be found.
pub fn selene_simple_runtime() -> Result<SeleneRuntime, RuntimeFetchError> {
    let runtime_path = find_built_selene_runtime("selene_simple_runtime")?;
    eprintln!(
        "[selene_simple_runtime] Found runtime at: {}",
        runtime_path.display()
    );
    let runtime = SeleneRuntime::new(runtime_path);
    Ok(runtime)
}

/// Create a Selene Soft RZ Runtime
///
/// This runtime implements soft RZ gates for more accurate gate modeling.
/// The runtime is expected to be at `../selene/target/release/libselene_soft_rz_runtime.so`
/// (relative to the PECOS workspace).
///
/// # Example
/// ```rust
/// use pecos_qis_selene::{selene_soft_rz_runtime};
/// use pecos_qis_core::{qis_engine, QisEngine};
/// use pecos_engines::ClassicalControlEngineBuilder;
/// use pecos_qis_ffi_types::OperationCollector;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Load the soft RZ runtime (built during compilation)
/// match selene_soft_rz_runtime() {
///     Ok(runtime) => {
///         let interface = OperationCollector::new();
///         let engine = qis_engine().runtime(runtime).program(interface).build()?;
///         // Engine is ready with soft RZ gate support
///     }
///     Err(e) => {
///         // Runtime not built - Selene repository not found
///         eprintln!("Soft RZ runtime not available: {}", e);
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Errors
/// Returns an error if the Selene soft RZ runtime library cannot be found.
pub fn selene_soft_rz_runtime() -> Result<SeleneRuntime, RuntimeFetchError> {
    let runtime_path = find_built_selene_runtime("selene_soft_rz_runtime")?;
    Ok(SeleneRuntime::new(runtime_path))
}

// Note: We only expose convenience functions for actual Selene runtime plugins.
// Other Selene plugins (error models, simulators, compilers) can still be loaded
// using find_selene_runtime() or selene_runtime() with an explicit path.

/// Find a Selene runtime that was built as a cargo dependency
///
/// This looks for the runtime libraries in the cargo target directory.
/// We search at runtime rather than using build-time environment variables because
/// the Selene runtimes are built as dependencies that may not exist when the build
/// script runs.
fn find_built_selene_runtime(lib_name: &str) -> Result<PathBuf, RuntimeFetchError> {
    // Platform-specific library extension
    let lib_ext = if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    };

    // Note: We don't check build-time environment variables here because they may be stale
    // The build script runs before Selene runtime dependencies are built, so those env vars
    // would point to non-existent paths. We rely solely on runtime detection instead.

    // Check cargo target directory for the dependency-built libraries
    // This handles the case where Selene runtimes are built as Cargo dependencies
    let target_dir = find_cargo_target_dir();
    if let Some(target) = target_dir {
        // Prefer the profile we're currently running in
        let current_profile = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        };
        let profiles = if current_profile == "release" {
            vec!["release", "debug"]
        } else {
            vec!["debug", "release"]
        };

        for profile in &profiles {
            // Check deps directory where cargo puts cdylib dependencies
            let deps_dir = target.join(profile).join("deps");
            if deps_dir.exists()
                && let Ok(entries) = std::fs::read_dir(&deps_dir)
            {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(filename) = path.file_name().and_then(|f| f.to_str())
                        // On Windows, libraries don't have "lib" prefix; on Unix they do
                        && (filename.starts_with(&format!("lib{lib_name}"))
                            || filename.starts_with(lib_name))
                        && path
                            .extension()
                            .is_some_and(|ext| ext.eq_ignore_ascii_case(lib_ext))
                    {
                        log::info!("Found Selene runtime in cargo deps: {}", path.display());
                        return Ok(path);
                    }
                }
            }

            // Also check standard location - try both with and without "lib" prefix
            let lib_prefix = if cfg!(target_os = "windows") {
                ""
            } else {
                "lib"
            };
            let runtime_path = target
                .join(profile)
                .join(format!("{lib_prefix}{lib_name}.{lib_ext}"));
            if runtime_path.exists() {
                log::info!(
                    "Found Selene runtime in cargo target: {}",
                    runtime_path.display()
                );
                return Ok(runtime_path);
            }
        }
    }

    Err(RuntimeFetchError::InvalidPath(format!(
        "Selene runtime {lib_name} not found. Make sure the selene-runtimes feature is enabled and the project is built."
    )))
}

/// Find the cargo target directory
fn find_cargo_target_dir() -> Option<PathBuf> {
    // First try CARGO_TARGET_DIR
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        return Some(PathBuf::from(target_dir));
    }

    // Otherwise look for target/ directory going up from current dir
    let mut current = std::env::current_dir().ok()?;
    loop {
        let target = current.join("target");
        if target.exists() && target.is_dir() {
            return Some(target);
        }
        if !current.pop() {
            break;
        }
    }

    None
}

/// Try to find a Selene runtime in common locations
///
/// Searches in order:
/// 1. `PECOS_SELENE_DIR` environment variable
/// 2. Current target/release or target/debug
/// 3. Workspace target directory
/// 4. System library paths
#[must_use]
pub fn find_selene_runtime(name: &str) -> Option<PathBuf> {
    // Platform-specific library extension
    let lib_ext = if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    };
    let filename = format!("libselene_{name}.{lib_ext}");

    // Check environment variable
    if let Ok(selene_dir) = std::env::var("PECOS_SELENE_DIR") {
        let path = PathBuf::from(selene_dir).join(&filename);
        if path.exists() {
            return Some(path);
        }
    }

    // Check target directories in current project
    for profile in &["release", "debug"] {
        let path = PathBuf::from("target").join(profile).join(&filename);
        if path.exists() {
            return Some(path);
        }

        // Check deps directory
        let deps_path = PathBuf::from("target").join(profile).join("deps");
        if deps_path.exists()
            && let Ok(entries) = std::fs::read_dir(&deps_path)
        {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(file_name) = path.file_name().and_then(|f| f.to_str())
                    && file_name.starts_with(&format!("libselene_{name}"))
                    && path
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case(lib_ext))
                {
                    return Some(path);
                }
            }
        }

        // Check parent directories (in case we're in a workspace member)
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let workspace_target = PathBuf::from(manifest_dir)
                .parent()?
                .parent()? // Go up to workspace root
                .join("target")
                .join(profile)
                .join(&filename);
            if workspace_target.exists() {
                return Some(workspace_target);
            }
        }
    }

    // Check system paths
    for sys_path in &["/usr/local/lib", "/usr/lib", "/opt/pecos/lib"] {
        let path = PathBuf::from(sys_path).join(&filename);
        if path.exists() {
            return Some(path);
        }
    }

    None
}

/// Create a Selene runtime automatically
///
/// This loads a runtime that was built by the build script. The name should be
/// the library name (e.g., "`selene_simple_runtime`", "`selene_soft_rz_runtime`").
///
/// # Example
/// ```rust
/// use pecos_qis_selene::{selene_runtime_auto};
/// use pecos_qis_core::{qis_engine, QisEngine};
/// use pecos_engines::ClassicalControlEngineBuilder;
/// use pecos_qis_ffi_types::OperationCollector;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Load a runtime by name (built during compilation)
/// match selene_runtime_auto("selene_simple_runtime") {
///     Ok(runtime) => {
///         let interface = OperationCollector::new();
///         let engine = qis_engine().runtime(runtime).program(interface).build()?;
///         // Engine is ready with the runtime
///     }
///     Err(e) => {
///         // Runtime not built - Selene repository not found
///         eprintln!("Could not load runtime: {}", e);
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Errors
/// Returns an error if the specified Selene runtime library cannot be found.
pub fn selene_runtime_auto(lib_name: &str) -> Result<SeleneRuntime, RuntimeFetchError> {
    let runtime_path = find_built_selene_runtime(lib_name)?;
    Ok(SeleneRuntime::new(runtime_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_selene_runtime() {
        // This might not find anything in test environment
        let result = find_selene_runtime("simple");
        // Just verify it doesn't panic
        if let Some(path) = result {
            assert!(path.to_string_lossy().contains("selene_simple"));
        }
    }
}
