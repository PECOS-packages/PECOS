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
use std::path::PathBuf;

/// Get the PECOS home directory
///
/// Returns `$PECOS_HOME` if set, otherwise `~/.pecos/`
///
/// # Errors
///
/// Returns an error if unable to determine the home directory
pub fn get_pecos_home() -> Result<PathBuf> {
    let home = if let Ok(dir) = std::env::var("PECOS_HOME") {
        PathBuf::from(dir)
    } else if let Some(home) = dirs::home_dir() {
        home.join(".pecos")
    } else {
        return Err(Error::HomeDir("Could not determine home directory".into()));
    };

    fs::create_dir_all(&home)?;
    Ok(home)
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
    let deps_dir = if let Ok(dir) = std::env::var("PECOS_DEPS_DIR") {
        PathBuf::from(dir)
    } else {
        get_pecos_home()?.join("deps")
    };

    fs::create_dir_all(&deps_dir)?;
    Ok(deps_dir)
}

/// Get the LLVM installation directory
///
/// Returns `$PECOS_HOME/llvm/`
///
/// # Errors
///
/// Returns an error if unable to determine or create the LLVM directory
pub fn get_llvm_dir() -> Result<PathBuf> {
    let llvm_dir = get_pecos_home()?.join("llvm");
    fs::create_dir_all(&llvm_dir)?;
    Ok(llvm_dir)
}

/// Get the cache directory for downloaded archives
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
    let cache_dir = if let Ok(dir) = std::env::var("PECOS_CACHE_DIR") {
        PathBuf::from(dir)
    } else {
        get_pecos_home()?.join("cache")
    };

    fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

/// Get the temporary directory for transient files during downloads/extraction
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
    let tmp_dir = get_pecos_home()?.join("tmp");
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

    // Note: These tests use unsafe env manipulation and must run with --test-threads=1

    #[test]
    fn test_get_pecos_home_default() {
        // SAFETY: Running with --test-threads=1, no concurrent access
        unsafe {
            std::env::remove_var("PECOS_HOME");
        }

        let home = get_pecos_home().expect("Should get PECOS home");
        assert!(home.ends_with(".pecos"), "Should end with .pecos");
        assert!(home.exists(), "Directory should be created");
    }

    #[test]
    fn test_get_deps_dir_default() {
        // SAFETY: Running with --test-threads=1, no concurrent access
        unsafe {
            std::env::remove_var("PECOS_HOME");
            std::env::remove_var("PECOS_DEPS_DIR");
        }

        let deps = get_deps_dir().expect("Should get deps dir");
        assert!(deps.ends_with("deps"), "Should end with deps");
        assert!(deps.exists(), "Directory should be created");
    }

    #[test]
    fn test_get_llvm_dir() {
        // SAFETY: Running with --test-threads=1, no concurrent access
        unsafe {
            std::env::remove_var("PECOS_HOME");
        }

        let llvm = get_llvm_dir().expect("Should get LLVM dir");
        assert!(llvm.ends_with("llvm"), "Should end with llvm");
        assert!(llvm.exists(), "Directory should be created");
    }

    #[test]
    fn test_get_cache_dir_default() {
        // SAFETY: Running with --test-threads=1, no concurrent access
        unsafe {
            std::env::remove_var("PECOS_HOME");
            std::env::remove_var("PECOS_CACHE_DIR");
        }

        let cache = get_cache_dir().expect("Should get cache dir");
        assert!(cache.ends_with("cache"), "Should end with cache");
        assert!(cache.exists(), "Directory should be created");
    }

    #[test]
    fn test_get_tmp_dir() {
        // SAFETY: Running with --test-threads=1, no concurrent access
        unsafe {
            std::env::remove_var("PECOS_HOME");
        }

        let tmp = get_tmp_dir().expect("Should get tmp dir");
        assert!(tmp.ends_with("tmp"), "Should end with tmp");
        assert!(tmp.exists(), "Directory should be created");
    }

    #[test]
    fn test_pecos_home_override() {
        let temp_dir = std::env::temp_dir().join("pecos_deps_test_home");
        // SAFETY: Running with --test-threads=1, no concurrent access
        unsafe {
            std::env::set_var("PECOS_HOME", &temp_dir);
        }

        let home = get_pecos_home().expect("Should get PECOS home");
        assert_eq!(home, temp_dir);
        assert!(home.exists(), "Directory should be created");

        // Cleanup
        // SAFETY: Running with --test-threads=1, no concurrent access
        unsafe {
            std::env::remove_var("PECOS_HOME");
        }
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
