//! cuTensor library management for PECOS
//!
//! cuTensor is a required runtime dependency of cuTensorNet (part of cuQuantum).
//! This module handles finding and installing cuTensor to `~/.pecos/deps/`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::errors::{Error, Result};
use crate::home::{get_cache_dir, get_deps_dir};

/// cuTensor version to install (major.minor.patch.build)
pub const CUTENSOR_VERSION: &str = "2.4.1.4";

/// Get the expected cuTensor directory name in deps
fn dep_dir_name() -> String {
    format!("cutensor-{CUTENSOR_VERSION}")
}

/// Find cuTensor installation
///
/// Search order:
/// 1. `~/.pecos/deps/cutensor-<version>/`
/// 2. System paths
#[must_use]
pub fn find_cutensor() -> Option<PathBuf> {
    // 1. Check ~/.pecos/deps/
    if let Ok(deps_dir) = get_deps_dir() {
        let path = deps_dir.join(dep_dir_name());
        if is_valid_cutensor(&path) {
            return Some(path);
        }
    }

    // 2. Check alongside cuQuantum installation
    if let Some(cuquantum_path) = crate::cuquantum::find_cuquantum()
        && has_cutensor_lib(&cuquantum_path)
    {
        return Some(cuquantum_path);
    }

    None
}

/// Check if a path contains cutensor libraries
#[must_use]
fn is_valid_cutensor(path: &Path) -> bool {
    has_cutensor_lib(path)
}

/// Check if libcutensor exists in a lib or lib64 subdirectory
#[must_use]
fn has_cutensor_lib(path: &Path) -> bool {
    for lib_dir in &["lib", "lib64"] {
        let lib = path.join(lib_dir).join("libcutensor.so");
        if lib.exists() {
            return true;
        }
        let lib_versioned = path.join(lib_dir).join("libcutensor.so.2");
        if lib_versioned.exists() {
            return true;
        }
    }
    false
}

/// Get the library directory within a cuTensor installation
#[must_use]
pub fn get_lib_dir(cutensor_path: &Path) -> Option<PathBuf> {
    let lib64 = cutensor_path.join("lib64");
    if lib64.exists() {
        return Some(lib64);
    }
    let lib = cutensor_path.join("lib");
    if lib.exists() {
        return Some(lib);
    }
    None
}

/// Install cuTensor to `~/.pecos/deps/cutensor-<version>/`
///
/// Downloads the cuTensor redistributable from NVIDIA and extracts it.
///
/// # Errors
///
/// Returns an error if download or extraction fails.
pub fn install_cutensor(force: bool) -> Result<PathBuf> {
    let deps_dir = get_deps_dir()?;
    let dest = deps_dir.join(dep_dir_name());

    if !force && is_valid_cutensor(&dest) {
        return Ok(dest);
    }

    if force && dest.exists() {
        fs::remove_dir_all(&dest)?;
    }

    let (url, filename) = get_download_info()?;
    let cache_dir = get_cache_dir()?;
    let archive_path = cache_dir.join(&filename);

    // Download if not cached
    if archive_path.exists() {
        println!(
            "cargo:warning=Using cached cuTensor download: {}",
            archive_path.display()
        );
    } else {
        println!("cargo:warning=Downloading cuTensor {CUTENSOR_VERSION}...");
        download(&url, &archive_path)?;
    }

    // Extract with --strip-components=1 into dest
    fs::create_dir_all(&dest)?;

    let status = Command::new("tar")
        .arg("-xf")
        .arg(&archive_path)
        .arg("-C")
        .arg(&dest)
        .arg("--strip-components=1")
        .status()
        .map_err(|e| Error::Archive(format!("Failed to run tar: {e}")))?;

    if !status.success() {
        return Err(Error::Archive("cuTensor extraction failed".into()));
    }

    if !is_valid_cutensor(&dest) {
        return Err(Error::Archive(
            "cuTensor extraction succeeded but no libraries found".into(),
        ));
    }

    println!(
        "cargo:warning=cuTensor {CUTENSOR_VERSION} installed to: {}",
        dest.display()
    );

    Ok(dest)
}

/// Ensure cuTensor is available, installing if needed
///
/// # Errors
///
/// Returns an error if cuTensor cannot be found or installed.
pub fn ensure_cutensor() -> Result<PathBuf> {
    if let Some(path) = find_cutensor() {
        return Ok(path);
    }
    install_cutensor(false)
}

/// Detect CUDA major version
fn detect_cuda_major() -> u32 {
    if let Some(cuda_path) = crate::cuda::find_cuda()
        && let Ok(version) = crate::cuda::get_cuda_version(&cuda_path)
        && let Some(major) = version.split('.').next()
        && let Ok(v) = major.parse::<u32>()
    {
        return v;
    }
    12
}

/// Get platform-specific download URL and filename
fn get_download_info() -> Result<(String, String)> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let cuda_major = detect_cuda_major();

    // cuTensor 2.3+ archives include _cuda12 or _cuda13 suffix
    match (os, arch) {
        ("linux", "x86_64") => Ok((
            format!(
                "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-{CUTENSOR_VERSION}_cuda{cuda_major}-archive.tar.xz"
            ),
            format!("libcutensor-linux-x86_64-{CUTENSOR_VERSION}_cuda{cuda_major}-archive.tar.xz"),
        )),
        ("linux", "aarch64") => Ok((
            format!(
                "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-sbsa/libcutensor-linux-sbsa-{CUTENSOR_VERSION}_cuda{cuda_major}-archive.tar.xz"
            ),
            format!("libcutensor-linux-sbsa-{CUTENSOR_VERSION}_cuda{cuda_major}-archive.tar.xz"),
        )),
        _ => Err(Error::CuQuantum(format!(
            "cuTensor is not available for {os}/{arch}"
        ))),
    }
}

/// Download a file from a URL, streaming to disk
fn download(url: &str, dest: &Path) -> Result<()> {
    let mut response = reqwest::blocking::get(url).map_err(|e| Error::Http(e.to_string()))?;

    if !response.status().is_success() {
        return Err(Error::Http(format!(
            "cuTensor download failed with status: {}",
            response.status()
        )));
    }

    let mut file = fs::File::create(dest)?;
    let bytes_copied = std::io::copy(&mut response, &mut file)
        .map_err(|e| Error::Http(format!("Failed to write download to disk: {e}")))?;

    println!(
        "cargo:warning=Downloaded cuTensor ({} MB)",
        bytes_copied / 1_000_000
    );

    Ok(())
}
