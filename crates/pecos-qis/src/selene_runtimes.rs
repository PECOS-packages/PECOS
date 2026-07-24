//! Utility functions for Selene runtime plugins
//!
//! Convenient access to Selene runtime implementations.
//! The runtimes are automatically built when you build this crate if the
//! Selene repository is found at ../selene (relative to PECOS).

use crate::SeleneRuntime;
use std::path::{Path, PathBuf};

fn platform_library_parts() -> (&'static str, &'static str) {
    if cfg!(target_os = "windows") {
        ("", "dll")
    } else if cfg!(target_os = "macos") {
        ("lib", "dylib")
    } else {
        ("lib", "so")
    }
}

pub(crate) fn platform_library_name(lib_name: &str) -> String {
    let (lib_prefix, lib_ext) = platform_library_parts();
    format!("{lib_prefix}{lib_name}.{lib_ext}")
}

pub(crate) fn platform_hashed_library_pattern(lib_name: &str) -> String {
    let (lib_prefix, lib_ext) = platform_library_parts();
    format!("{lib_prefix}{lib_name}-*.{lib_ext}")
}

/// Find an exact or Cargo hash-suffixed dynamic library in a directory or its `deps/`.
pub(crate) fn find_library_in_dir(dir: &Path, lib_name: &str) -> Option<PathBuf> {
    let (lib_prefix, lib_ext) = platform_library_parts();
    let exact_name = platform_library_name(lib_name);
    let hashed_stem_prefix = format!("{lib_prefix}{lib_name}-");

    for search_dir in [dir.to_path_buf(), dir.join("deps")] {
        let exact_path = search_dir.join(&exact_name);
        if exact_path.is_file() {
            return Some(exact_path);
        }

        let mut hashed_paths = std::fs::read_dir(&search_dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file()
                    && path
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case(lib_ext))
                    && path
                        .file_stem()
                        .and_then(|stem| stem.to_str())
                        .and_then(|stem| stem.strip_prefix(&hashed_stem_prefix))
                        .is_some_and(|hash| !hash.is_empty())
            })
            .collect::<Vec<_>>();
        hashed_paths.sort_unstable();
        if let Some(path) = hashed_paths.into_iter().next() {
            return Some(path);
        }
    }

    None
}

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
/// use pecos_qis::selene_simple_runtime;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Load the simple runtime (built during compilation)
/// match selene_simple_runtime() {
///     Ok(runtime) => {
///         println!("Runtime loaded successfully");
///         // Use with qis_engine().runtime(runtime).interface(...).program(...).build()
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
    log::debug!(
        "selene_simple_runtime: Found runtime at: {}",
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
/// use pecos_qis::selene_soft_rz_runtime;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Load the soft RZ runtime (built during compilation)
/// match selene_soft_rz_runtime() {
///     Ok(runtime) => {
///         println!("Soft RZ runtime loaded successfully");
///         // Use with qis_engine().runtime(runtime).interface(...).program(...).build()
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
            ["release", "debug"]
        } else {
            ["debug", "release"]
        };

        for profile in &profiles {
            if let Some(runtime_path) = find_library_in_dir(&target.join(profile), lib_name) {
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

    // Development wheels and editable installs may run from an arbitrary cwd.
    // The compile-time crate path still identifies this workspace checkout.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(workspace_root) = manifest_dir.parent().and_then(std::path::Path::parent) {
        let target = workspace_root.join("target");
        if target.is_dir() {
            return Some(target);
        }
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
    let lib_name = if name.starts_with("selene_") {
        name.to_string()
    } else {
        format!("selene_{name}")
    };
    let filename = platform_library_name(&lib_name);
    let hashed_pattern = platform_hashed_library_pattern(&lib_name);

    // Check environment variable
    if let Some(selene_dir) = std::env::var_os("PECOS_SELENE_DIR") {
        let selene_dir = PathBuf::from(selene_dir);
        if let Some(path) = find_library_in_dir(&selene_dir, &lib_name) {
            return Some(path);
        }
        log::warn!(
            "PECOS_SELENE_DIR is set to {}, but neither {filename} nor {hashed_pattern} was found there or in its deps directory",
            selene_dir.display()
        );
    }

    // Check target directories in current project
    for profile in &["release", "debug"] {
        let profile_dir = PathBuf::from("target").join(profile);
        if let Some(path) = find_library_in_dir(&profile_dir, &lib_name) {
            return Some(path);
        }

        // Check parent directories (in case we're in a workspace member)
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let workspace_profile = PathBuf::from(manifest_dir)
                .parent()?
                .parent()? // Go up to workspace root
                .join("target")
                .join(profile);
            if let Some(path) = find_library_in_dir(&workspace_profile, &lib_name) {
                return Some(path);
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
/// use pecos_qis::selene_runtime_auto;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Load a runtime by name (built during compilation)
/// match selene_runtime_auto("selene_simple_runtime") {
///     Ok(runtime) => {
///         println!("Runtime loaded successfully");
///         // Use with qis_engine().runtime(runtime).interface(...).program(...).build()
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
    use crate::test_env::{ENV_MUTEX, EnvVarGuard};
    use std::fs::File;

    #[test]
    fn test_find_selene_runtime() {
        let _env_lock = ENV_MUTEX
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // This might not find anything in test environment
        let result = find_selene_runtime("simple");
        // Just verify it doesn't panic
        if let Some(path) = result {
            assert!(path.to_string_lossy().contains("selene_simple"));
        }
    }

    #[test]
    fn test_find_selene_runtime_in_hashed_env_deps_dir() {
        let _env_lock = ENV_MUTEX
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let temp_dir = tempfile::tempdir().expect("create temp directory");
        let deps_dir = temp_dir.path().join("deps");
        std::fs::create_dir(&deps_dir).expect("create deps directory");
        let runtime_path = deps_dir.join({
            let (prefix, ext) = platform_library_parts();
            format!("{prefix}selene_simple_runtime-abc123.{ext}")
        });
        File::create(&runtime_path).expect("create hashed runtime library");
        let _env = EnvVarGuard::set("PECOS_SELENE_DIR", temp_dir.path());

        assert_eq!(
            find_selene_runtime("selene_simple_runtime"),
            Some(runtime_path)
        );
    }
}
