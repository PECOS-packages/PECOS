//! CUDA Toolkit management for PECOS
//!
//! This module provides functionality to download, install, and manage
//! CUDA Toolkit installations in `~/.pecos/cuda/`.

pub mod installer;

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::errors::{Error, Result};

/// CUDA version we install
pub const CUDA_VERSION: &str = "12.6.3";

/// Get the pecos CUDA installation directory
#[must_use]
pub fn get_pecos_cuda_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".pecos").join("cuda"))
}

/// Find CUDA installation, checking local first, then system
///
/// Search order:
/// 1. `~/.pecos/cuda/` (local installation)
/// 2. `CUDA_PATH` environment variable
/// 3. `nvcc` in PATH (derive `CUDA_PATH` from nvcc location)
/// 4. Standard system paths (`/usr/local/cuda`, etc.)
#[must_use]
pub fn find_cuda() -> Option<PathBuf> {
    // 1. Check ~/.pecos/cuda/ first (local installation)
    if let Some(pecos_cuda) = get_pecos_cuda_dir()
        && is_valid_cuda_installation(&pecos_cuda)
    {
        return Some(pecos_cuda);
    }

    // 2. Check CUDA_PATH environment variable
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let path = PathBuf::from(&cuda_path);
        if is_valid_cuda_installation(&path) {
            return Some(path);
        }
    }

    // 3. Try to find nvcc in PATH and derive CUDA_PATH
    if let Ok(output) = Command::new("which").arg("nvcc").output()
        && output.status.success()
    {
        let nvcc_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        // nvcc is typically at /usr/local/cuda/bin/nvcc
        // CUDA_PATH would be /usr/local/cuda
        if let Some(cuda_path) = PathBuf::from(&nvcc_path)
                .parent() // bin/
                .and_then(|p| p.parent()) // cuda/
                && is_valid_cuda_installation(cuda_path)
        {
            return Some(cuda_path.to_path_buf());
        }
    }

    // 4. Check standard system paths
    let standard_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12",
        "/opt/cuda",
    ];

    for path_str in &standard_paths {
        let path = PathBuf::from(path_str);
        if is_valid_cuda_installation(&path) {
            return Some(path);
        }
    }

    None
}

/// Check if a path contains a valid CUDA installation
#[must_use]
pub fn is_valid_cuda_installation(path: &Path) -> bool {
    if !path.exists() {
        return false;
    }

    let exe_ext = if cfg!(windows) { ".exe" } else { "" };

    // Check for nvcc
    let nvcc = path.join("bin").join(format!("nvcc{exe_ext}"));
    if !nvcc.exists() {
        return false;
    }

    // Check for cuda_runtime.h
    let runtime_header = path.join("include").join("cuda_runtime.h");
    if !runtime_header.exists() {
        return false;
    }

    true
}

/// Get CUDA version from an installation
///
/// # Errors
/// Returns an error if nvcc cannot be executed or version cannot be parsed.
pub fn get_cuda_version(cuda_path: &Path) -> Result<String> {
    let exe_ext = if cfg!(windows) { ".exe" } else { "" };
    let nvcc = cuda_path.join("bin").join(format!("nvcc{exe_ext}"));

    let output = Command::new(&nvcc)
        .arg("--version")
        .output()
        .map_err(|e| Error::Cuda(format!("Failed to execute nvcc: {e}")))?;

    if !output.status.success() {
        return Err(Error::Cuda("nvcc --version failed".into()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse version from output like "Cuda compilation tools, release 12.6, V12.6.77"
    stdout
        .lines()
        .find(|l| l.contains("release"))
        .and_then(|l| l.split("release ").nth(1))
        .map(|s| s.split(',').next().unwrap_or(s).trim().to_string())
        .ok_or_else(|| Error::Cuda("Could not parse CUDA version from nvcc output".into()))
}

/// Check if CUDA is available (either local or system)
#[must_use]
pub fn is_cuda_available() -> bool {
    find_cuda().is_some()
}
