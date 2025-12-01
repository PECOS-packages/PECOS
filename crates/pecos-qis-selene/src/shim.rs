//! Selene Runtime Shim
//!
//! This module provides the C shim library that implements selene_* functions
//! and forwards them to PECOS's thread-local interface.
//!
//! The shim is compiled as a shared library (`libpecos_selene_shim.so`) that
//! provides the selene_* symbols expected by libhelios.a.

// The actual shim is implemented in C (src/c/selene_shim.c)
// This module just provides Rust-side utilities if needed

use log::debug;
use std::path::{Path, PathBuf};

/// Get the library name for the current platform
fn shim_lib_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "libpecos_selene.dylib"
    } else if cfg!(target_os = "windows") {
        "pecos_selene.dll"
    } else {
        "libpecos_selene.so"
    }
}

/// Derive the project target directory from the compile-time embedded path.
///
/// The compile-time path looks like:
/// `/path/to/project/target/release/build/pecos-qis-selene-HASH/out/libpecos_selene.so`
///
/// We want to extract `/path/to/project/target` so we can search for other build hashes.
fn get_project_target_dir() -> Option<PathBuf> {
    let compile_time_path = PathBuf::from(env!("PECOS_SELENE_SHIM_PATH"));
    // Go up from: libpecos_selene.so -> out -> pecos-qis-selene-HASH -> build -> release/debug -> target
    compile_time_path
        .parent() // out/
        .and_then(|p| p.parent()) // pecos-qis-selene-HASH/
        .and_then(|p| p.parent()) // build/
        .and_then(|p| p.parent()) // release or debug
        .and_then(|p| p.parent()) // target/
        .map(std::path::Path::to_path_buf)
}

/// Search for the shim library in a target directory
fn search_target_dir(target_dir: &Path, lib_name: &str) -> Option<PathBuf> {
    for profile in ["release", "debug"] {
        let build_dir = target_dir.join(profile).join("build");
        if build_dir.exists()
            && let Ok(entries) = std::fs::read_dir(&build_dir)
        {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("pecos-qis-selene-") {
                    let lib_path = entry.path().join("out").join(lib_name);
                    if lib_path.exists() {
                        debug!("Found PECOS shim library at: {}", lib_path.display());
                        return Some(lib_path);
                    }
                }
            }
        }
    }
    None
}

/// Get the path to the compiled shim library
///
/// The shim is compiled by build.rs and placed in the output directory.
/// Search order:
/// 1. Runtime `PECOS_SELENE_SHIM_PATH` environment variable (explicit override)
/// 2. Embedded path from build time (compile-time `PECOS_SELENE_SHIM_PATH`)
/// 3. Search target directory derived from compile-time path (handles hash changes)
/// 4. Search target directory relative to current working directory
#[must_use]
pub fn get_shim_library_path() -> Option<PathBuf> {
    let lib_name = shim_lib_name();

    // 1. Check runtime environment variable (explicit override)
    if let Ok(path_str) = std::env::var("PECOS_SELENE_SHIM_PATH") {
        let path = PathBuf::from(&path_str);
        if path.exists() {
            debug!(
                "Using PECOS shim library from PECOS_SELENE_SHIM_PATH env var: {}",
                path.display()
            );
            return Some(path);
        }
    }

    // 2. Check compile-time embedded path
    let compile_time_path = PathBuf::from(env!("PECOS_SELENE_SHIM_PATH"));
    if compile_time_path.exists() {
        debug!(
            "Using PECOS shim library from compile-time path: {}",
            compile_time_path.display()
        );
        return Some(compile_time_path);
    }

    // 3. Search target directory derived from compile-time path
    // This handles cases where the build hash changed but the target dir is the same
    if let Some(target_dir) = get_project_target_dir()
        && let Some(path) = search_target_dir(&target_dir, lib_name)
    {
        return Some(path);
    }

    // 4. Search target directory relative to current working directory
    if let Ok(cwd) = std::env::current_dir() {
        let target_dir = cwd.join("target");
        if let Some(path) = search_target_dir(&target_dir, lib_name) {
            return Some(path);
        }
    }

    // Nothing found
    None
}
