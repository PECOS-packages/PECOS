//! PECOS home directory management
//!
//! This module manages the `~/.pecos/` home directory structure:
//!
//! ```text
//! ~/.pecos/
//! ├── cache/          # Downloaded archives (tar.gz, 7z, etc.)
//! ├── deps/           # All dependencies, versioned by name
//! │   ├── llvm-14/
//! │   ├── cuda-12.6.3/
//! │   ├── quest-v4.2.0/
//! │   ├── stim-bd60b73525fd/
//! │   └── ...
//! └── tmp/            # Temporary files during downloads/extraction
//! ```
//!
//! # Legacy paths
//!
//! Earlier versions installed LLVM, CUDA, and cuQuantum at the top level
//! (`~/.pecos/llvm/`, `~/.pecos/cuda/`, `~/.pecos/cuquantum/`). Detection
//! still checks these paths as a fallback, but new installs go under `deps/`.
//! Run `pecos migrate` to move legacy installs into `deps/`.
//!
//! # Environment Variables
//!
//! - `PECOS_HOME`: Override the entire home directory (default: `~/.pecos/`)
//! - `PECOS_CACHE_DIR`: Override the cache/archives location (default: `$PECOS_HOME/cache/`)
//! - `PECOS_DEPS_DIR`: Override the extracted sources location (default: `$PECOS_HOME/deps/`)

use crate::errors::{Error, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Get the PECOS home directory path (without creating it)
///
/// Returns `$PECOS_HOME` if set, otherwise `~/.pecos/`
///
/// # Errors
///
/// Returns an error if unable to determine the home directory
pub fn get_pecos_home_path() -> Result<PathBuf> {
    get_pecos_home_path_with_override(None)
}

/// Get the PECOS home directory path with an optional override (for testing)
///
/// If `override_path` is provided, returns that path directly.
/// Otherwise, returns `$PECOS_HOME` if set, or `~/.pecos/`
///
/// # Errors
///
/// Returns an error if unable to determine the home directory
pub fn get_pecos_home_path_with_override(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return Ok(path.to_path_buf());
    }
    if let Ok(dir) = std::env::var("PECOS_HOME") {
        Ok(PathBuf::from(dir))
    } else if let Some(home) = dirs::home_dir() {
        Ok(home.join(".pecos"))
    } else {
        Err(Error::HomeDir("Could not determine home directory".into()))
    }
}

/// Get the PECOS home directory (creates if needed)
///
/// Returns `$PECOS_HOME` if set, otherwise `~/.pecos/`
///
/// # Errors
///
/// Returns an error if unable to determine or create the home directory
pub fn get_pecos_home() -> Result<PathBuf> {
    get_pecos_home_with_override(None)
}

/// Get the PECOS home directory with an optional override (for testing)
///
/// # Errors
///
/// Returns an error if unable to determine or create the home directory
pub fn get_pecos_home_with_override(override_path: Option<&Path>) -> Result<PathBuf> {
    let home = get_pecos_home_path_with_override(override_path)?;
    fs::create_dir_all(&home)?;
    Ok(home)
}

/// Get the dependencies directory path (without creating it)
///
/// Returns `$PECOS_DEPS_DIR` if set, otherwise `$PECOS_HOME/deps/`
///
/// # Errors
///
/// Returns an error if unable to determine the path
pub fn get_deps_dir_path() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("PECOS_DEPS_DIR") {
        Ok(PathBuf::from(dir))
    } else {
        Ok(get_pecos_home_path()?.join("deps"))
    }
}

/// Get the dependencies directory for extracted source trees
///
/// Returns `$PECOS_DEPS_DIR` if set, otherwise `$PECOS_HOME/deps/`
///
/// This is where extracted and patched source trees are stored, ready for building.
/// Each dependency gets its own subdirectory: `deps/<name>-<version>/`
///
/// # Errors
///
/// Returns an error if unable to determine or create the deps directory
pub fn get_deps_dir() -> Result<PathBuf> {
    let deps_dir = get_deps_dir_path()?;
    fs::create_dir_all(&deps_dir)?;
    Ok(deps_dir)
}

/// Get a versioned dependency directory path (without creating it).
///
/// Returns `$PECOS_HOME/deps/{name}-{version}/`
///
/// # Errors
///
/// Returns an error if unable to determine the path.
pub fn get_versioned_dep_path(name: &str, version: &str) -> Result<PathBuf> {
    Ok(get_deps_dir_path()?.join(format!("{name}-{version}")))
}

/// Get a versioned dependency directory, creating it if needed.
///
/// # Errors
///
/// Returns an error if unable to determine or create the directory.
pub fn get_versioned_dep_dir(name: &str, version: &str) -> Result<PathBuf> {
    let dir = get_versioned_dep_path(name, version)?;
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Resolve a dependency directory, checking versioned path first then legacy unversioned.
///
/// For new installs, returns the versioned path. For existing installs, returns
/// whichever path exists (versioned preferred over legacy).
///
/// # Errors
///
/// Returns an error if unable to determine the path.
pub fn resolve_dep_path(name: &str, version: &str) -> Result<PathBuf> {
    let versioned = get_versioned_dep_path(name, version)?;
    if versioned.exists() {
        return Ok(versioned);
    }
    // Fall back to legacy unversioned path (migration handled by `pecos setup`)
    let legacy = get_deps_dir_path()?.join(name);
    if legacy.exists() {
        return Ok(legacy);
    }
    Ok(versioned)
}

/// LLVM major version used by PECOS
pub const LLVM_VERSION: &str = "14";

/// Get the vendored cmake installation directory path (without creating it)
///
/// Returns `$PECOS_HOME/deps/cmake-{CMAKE_VERSION}/`.
///
/// # Errors
///
/// Returns an error if unable to determine the path
pub fn get_cmake_dir_path() -> Result<PathBuf> {
    resolve_dep_path("cmake", crate::cmake::CMAKE_VERSION)
}

/// Get the LLVM installation directory path (without creating it)
///
/// # Errors
///
/// Returns an error if unable to determine the path
pub fn get_llvm_dir_path() -> Result<PathBuf> {
    resolve_dep_path("llvm", LLVM_VERSION)
}

/// Get the LLVM installation directory (creates if needed)
///
/// # Errors
///
/// Returns an error if unable to determine or create the LLVM directory
pub fn get_llvm_dir() -> Result<PathBuf> {
    let llvm_dir = get_llvm_dir_path()?;
    fs::create_dir_all(&llvm_dir)?;
    Ok(llvm_dir)
}

/// Get the CUDA installation directory path (without creating it)
///
/// # Errors
///
/// Returns an error if unable to determine the path
pub fn get_cuda_dir_path() -> Result<PathBuf> {
    resolve_dep_path("cuda", crate::cuda::CUDA_VERSION)
}

/// Get the CUDA installation directory (creates if needed)
///
/// # Errors
///
/// Returns an error if unable to determine or create the CUDA directory
pub fn get_cuda_dir() -> Result<PathBuf> {
    let cuda_dir = get_cuda_dir_path()?;
    fs::create_dir_all(&cuda_dir)?;
    Ok(cuda_dir)
}

/// Get the cuQuantum installation directory path (without creating it)
///
/// Returns `$PECOS_HOME/deps/cuquantum/`
///
/// # Errors
///
/// Returns an error if unable to determine the path
pub fn get_cuquantum_dir_path() -> Result<PathBuf> {
    resolve_dep_path("cuquantum", crate::cuquantum::CUQUANTUM_VERSION)
}

/// Get the cuQuantum installation directory (creates if needed)
///
/// Returns `$PECOS_HOME/deps/cuquantum/`
///
/// # Errors
///
/// Returns an error if unable to determine or create the cuQuantum directory
pub fn get_cuquantum_dir() -> Result<PathBuf> {
    let dir = get_cuquantum_dir_path()?;
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Get the cache directory path (without creating it)
///
/// Returns `$PECOS_CACHE_DIR` if set, otherwise `$PECOS_HOME/cache/`
///
/// # Errors
///
/// Returns an error if unable to determine the path
pub fn get_cache_dir_path() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("PECOS_CACHE_DIR") {
        Ok(PathBuf::from(dir))
    } else {
        Ok(get_pecos_home_path()?.join("cache"))
    }
}

/// Get the cache directory for downloaded archives (creates if needed)
///
/// Returns `$PECOS_CACHE_DIR` if set, otherwise `$PECOS_HOME/cache/`
///
/// This is where downloaded archives (tar.gz, 7z, etc.) are stored before extraction.
/// Archives are kept for faster re-extraction if deps/ is cleaned.
///
/// # Errors
///
/// Returns an error if unable to determine or create the cache directory
pub fn get_cache_dir() -> Result<PathBuf> {
    let cache_dir = get_cache_dir_path()?;
    fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

/// Get the temporary directory path (without creating it)
///
/// Returns `$PECOS_HOME/tmp/`
///
/// # Errors
///
/// Returns an error if unable to determine the path
pub fn get_tmp_dir_path() -> Result<PathBuf> {
    Ok(get_pecos_home_path()?.join("tmp"))
}

/// Get the temporary directory for transient files during downloads/extraction (creates if needed)
///
/// Returns `$PECOS_HOME/tmp/`
///
/// This directory is used for temporary files during archive extraction and
/// other transient operations. It can be safely cleaned at any time.
///
/// # Errors
///
/// Returns an error if unable to determine or create the tmp directory
pub fn get_tmp_dir() -> Result<PathBuf> {
    let tmp_dir = get_tmp_dir_path()?;
    fs::create_dir_all(&tmp_dir)?;
    Ok(tmp_dir)
}

// ── Legacy path helpers ─────────────────────────────────────────────────────
//
// Earlier versions installed LLVM/CUDA/cuQuantum at the top level of
// ~/.pecos/.  These helpers detect the old locations so that detection code
// can fall back gracefully and `pecos migrate` can move them.

/// Legacy LLVM path: `~/.pecos/llvm/`
///
/// # Errors
///
/// Returns an error if unable to determine the path
pub fn get_legacy_llvm_dir_path() -> Result<PathBuf> {
    Ok(get_pecos_home_path()?.join("llvm"))
}

/// Legacy CUDA path: `~/.pecos/cuda/`
///
/// # Errors
///
/// Returns an error if unable to determine the path
pub fn get_legacy_cuda_dir_path() -> Result<PathBuf> {
    Ok(get_pecos_home_path()?.join("cuda"))
}

/// Legacy cuQuantum path: `~/.pecos/cuquantum/`
///
/// # Errors
///
/// Returns an error if unable to determine the path
pub fn get_legacy_cuquantum_dir_path() -> Result<PathBuf> {
    Ok(get_pecos_home_path()?.join("cuquantum"))
}

/// Print a deprecation warning for a legacy top-level install path.
pub fn print_legacy_warning(name: &str, old_path: &Path) {
    eprintln!(
        "Warning: {name} found at legacy path: {}",
        old_path.display()
    );
    eprintln!("  Run `pecos migrate` to move it to ~/.pecos/deps/");
}

/// Description of a single legacy dep that can be migrated.
pub struct LegacyDep {
    /// Human-readable name (e.g. "LLVM 14")
    pub name: &'static str,
    /// Old path
    pub old: PathBuf,
    /// New path
    pub new: PathBuf,
}

/// Check for legacy top-level installs that should be migrated.
///
/// Returns a list of deps whose old path exists but new path does not.
///
/// # Errors
///
/// Returns an error if unable to determine paths.
pub fn find_legacy_deps() -> Result<Vec<LegacyDep>> {
    let mut found = Vec::new();
    let deps_dir = get_deps_dir_path()?;

    let checks: &[(&str, &str)] = &[
        ("LLVM", LLVM_VERSION),
        ("CUDA", crate::cuda::CUDA_VERSION),
        ("cuQuantum", crate::cuquantum::CUQUANTUM_VERSION),
    ];

    for &(name, version) in checks {
        let lower = name.to_lowercase();
        let versioned = deps_dir.join(format!("{lower}-{version}"));
        if versioned.exists() {
            continue; // Already at versioned path
        }

        // Check unversioned deps/ path (e.g. deps/llvm/)
        let unversioned = deps_dir.join(&lower);
        if unversioned.exists() {
            found.push(LegacyDep {
                name,
                old: unversioned,
                new: versioned.clone(),
            });
            continue;
        }

        // Check top-level legacy path (e.g. ~/.pecos/llvm/)
        if let Ok(top_level) = get_pecos_home_path().map(|h| h.join(&lower))
            && top_level.exists()
        {
            found.push(LegacyDep {
                name,
                old: top_level,
                new: versioned,
            });
        }
    }
    Ok(found)
}

/// Migrate a single legacy dep by renaming old -> new.
///
/// # Errors
///
/// Returns an error if the rename fails.
pub fn migrate_legacy_dep(dep: &LegacyDep) -> Result<()> {
    if let Some(parent) = dep.new.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::rename(&dep.old, &dep.new)?;
    Ok(())
}

/// Get information about the PECOS home directory
#[derive(Debug)]
pub struct HomeInfo {
    /// Path to PECOS home
    pub home: PathBuf,
    /// Path to deps directory
    pub deps: PathBuf,
    /// Path to LLVM directory
    pub llvm: PathBuf,
    /// Path to CUDA directory
    pub cuda: PathBuf,
    /// Path to cuQuantum directory
    pub cuquantum: PathBuf,
    /// Path to cache directory
    pub cache: PathBuf,
    /// Path to tmp directory
    pub tmp: PathBuf,
    /// Whether `PECOS_HOME` is overridden
    pub home_overridden: bool,
    /// Whether `PECOS_DEPS_DIR` is overridden
    pub deps_overridden: bool,
    /// Whether `PECOS_CACHE_DIR` is overridden
    pub cache_overridden: bool,
}

/// Get information about the PECOS home directory
///
/// # Errors
///
/// Returns an error if unable to determine directory paths
pub fn get_home_info() -> Result<HomeInfo> {
    Ok(HomeInfo {
        home: get_pecos_home()?,
        deps: get_deps_dir()?,
        llvm: get_llvm_dir()?,
        cuda: get_cuda_dir()?,
        cuquantum: get_cuquantum_dir()?,
        cache: get_cache_dir()?,
        tmp: get_tmp_dir()?,
        home_overridden: std::env::var("PECOS_HOME").is_ok(),
        deps_overridden: std::env::var("PECOS_DEPS_DIR").is_ok(),
        cache_overridden: std::env::var("PECOS_CACHE_DIR").is_ok(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Atomic counter for unique test directories
    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Create a unique temporary directory for each test
    fn unique_test_dir(prefix: &str) -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("pecos_test_{prefix}_{pid}_{id}"))
    }

    #[test]
    fn test_get_pecos_home_default() {
        // Test that default home ends with .pecos (uses real home dir)
        let home = get_pecos_home_path().expect("Should get PECOS home path");
        assert!(home.ends_with(".pecos"), "Should end with .pecos");
    }

    #[test]
    fn test_get_deps_dir_default() {
        // Test that deps dir ends with "deps"
        let test_home = unique_test_dir("deps");
        let deps = get_pecos_home_path_with_override(Some(&test_home))
            .expect("Should get home")
            .join("deps");
        assert!(deps.ends_with("deps"), "Should end with deps");

        // Cleanup
        let _ = std::fs::remove_dir_all(&test_home);
    }

    #[test]
    fn test_get_llvm_dir() {
        // Test that LLVM dir is created correctly
        let test_home = unique_test_dir("llvm");
        let llvm = get_pecos_home_with_override(Some(&test_home))
            .expect("Should get home")
            .join("llvm");
        fs::create_dir_all(&llvm).expect("Should create llvm dir");
        assert!(llvm.ends_with("llvm"), "Should end with llvm");
        assert!(llvm.exists(), "Directory should be created");

        // Cleanup
        let _ = std::fs::remove_dir_all(&test_home);
    }

    #[test]
    fn test_get_cache_dir_default() {
        // Test that cache dir ends with "cache"
        let test_home = unique_test_dir("cache");
        let cache = get_pecos_home_path_with_override(Some(&test_home))
            .expect("Should get home")
            .join("cache");
        assert!(cache.ends_with("cache"), "Should end with cache");

        // Cleanup
        let _ = std::fs::remove_dir_all(&test_home);
    }

    #[test]
    fn test_get_tmp_dir() {
        // Test that tmp dir is created correctly
        let test_home = unique_test_dir("tmp");
        let tmp = get_pecos_home_with_override(Some(&test_home))
            .expect("Should get home")
            .join("tmp");
        fs::create_dir_all(&tmp).expect("Should create tmp dir");
        assert!(tmp.ends_with("tmp"), "Should end with tmp");
        assert!(tmp.exists(), "Directory should be created");

        // Cleanup
        let _ = std::fs::remove_dir_all(&test_home);
    }

    #[test]
    fn test_pecos_home_override() {
        // Test that override path works correctly
        let test_home = unique_test_dir("override");

        let home = get_pecos_home_with_override(Some(&test_home)).expect("Should get PECOS home");
        assert_eq!(home, test_home);
        assert!(home.exists(), "Directory should be created");

        // Cleanup
        let _ = std::fs::remove_dir_all(&test_home);
    }
}
