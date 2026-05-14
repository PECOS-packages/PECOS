//! cmake detection and vendored install.
//!
//! cmake is required by the optional MWPF decoder (via the highs-sys C++ LP
//! solver). PECOS handles cmake the same way it handles LLVM: a system install
//! is the fast path, and `pecos install cmake` will fetch a vendored copy from
//! Kitware into `~/.pecos/deps/cmake-{CMAKE_VERSION}/` when none is found.

pub mod installer;

use std::path::{Path, PathBuf};
use std::process::Command;

/// cmake version PECOS vendors. Pinned to the latest 3.x release so that
/// projects with `cmake_minimum_required(VERSION 3.x)` (like `HiGHS`) keep
/// building without policy compatibility surprises from cmake 4.x.
pub const CMAKE_VERSION: &str = "3.31.12";

/// Find a usable cmake.
///
/// Search order:
/// 1. PECOS-managed install at `~/.pecos/deps/cmake-{CMAKE_VERSION}/`
/// 2. `cmake` on PATH
///
/// Returns `Some(path-to-cmake-binary)` or `None`.
#[must_use]
pub fn find_cmake() -> Option<PathBuf> {
    if let Ok(vendored_root) = crate::home::get_cmake_dir_path()
        && let Some(bin) = cmake_binary_in(&vendored_root)
        && bin.is_file()
    {
        return Some(bin);
    }
    find_system_cmake()
}

/// Find cmake on the system PATH.
#[must_use]
pub fn find_system_cmake() -> Option<PathBuf> {
    let output = Command::new("cmake").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    which_in_path("cmake")
}

/// Directory containing the cmake binary for a given installation root.
///
/// On Linux and Windows the layout is `{root}/bin/cmake[.exe]`. On macOS the
/// upstream tarball nests the binary inside `CMake.app/Contents/bin/`.
#[must_use]
pub fn cmake_bin_dir(root: &Path) -> PathBuf {
    if cfg!(target_os = "macos") {
        root.join("CMake.app").join("Contents").join("bin")
    } else {
        root.join("bin")
    }
}

/// Path to the cmake binary inside an installation root, if it exists.
#[must_use]
pub fn cmake_binary_in(root: &Path) -> Option<PathBuf> {
    let bin_name = if cfg!(windows) { "cmake.exe" } else { "cmake" };
    let candidate = cmake_bin_dir(root).join(bin_name);
    candidate.is_file().then_some(candidate)
}

fn which_in_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    let exts: &[&str] = if cfg!(windows) {
        &[".exe", ".bat", ""]
    } else {
        &[""]
    };
    for dir in std::env::split_paths(&path_var) {
        for ext in exts {
            let candidate = dir.join(format!("{name}{ext}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// The docs URL we point users at for manual install instructions.
pub const DOCS_URL: &str =
    "https://github.com/PECOS-packages/PECOS/blob/dev/docs/user-guide/cmake-setup.md";
