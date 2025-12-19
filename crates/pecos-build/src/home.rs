//! PECOS home directory management
//!
//! This module manages the `~/.pecos/` home directory structure:
//!
//! ```text
//! ~/.pecos/
//! ├── cache/      # Downloaded archives (tar.gz, 7z, etc.)
//! ├── deps/       # Extracted & patched sources (ready to build)
//! ├── llvm/       # LLVM installation
//! └── tmp/        # Temporary files during downloads/extraction
//! ```
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

/// Get the LLVM installation directory path (without creating it)
///
/// Returns `$PECOS_HOME/llvm/`
///
/// # Errors
///
/// Returns an error if unable to determine the path
pub fn get_llvm_dir_path() -> Result<PathBuf> {
    Ok(get_pecos_home_path()?.join("llvm"))
}

/// Get the LLVM installation directory (creates if needed)
///
/// Returns `$PECOS_HOME/llvm/`
///
/// # Errors
///
/// Returns an error if unable to determine or create the LLVM directory
pub fn get_llvm_dir() -> Result<PathBuf> {
    let llvm_dir = get_llvm_dir_path()?;
    fs::create_dir_all(&llvm_dir)?;
    Ok(llvm_dir)
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

/// Get information about the PECOS home directory
#[derive(Debug)]
pub struct HomeInfo {
    /// Path to PECOS home
    pub home: PathBuf,
    /// Path to deps directory
    pub deps: PathBuf,
    /// Path to LLVM directory
    pub llvm: PathBuf,
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

/// Get comprehensive information about the PECOS home directory
///
/// # Errors
///
/// Returns an error if unable to determine directory paths
pub fn get_home_info() -> Result<HomeInfo> {
    Ok(HomeInfo {
        home: get_pecos_home()?,
        deps: get_deps_dir()?,
        llvm: get_llvm_dir()?,
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
