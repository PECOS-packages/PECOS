//! cuQuantum SDK management for PECOS
//!
//! This module provides functionality to detect, download, and manage
//! cuQuantum SDK installations in `~/.pecos/cuquantum/`.
//!
//! cuQuantum is NVIDIA's SDK for accelerated quantum circuit simulation,
//! including cuStateVec (state vector) and cuTensorNet (tensor network).

pub mod config;
pub mod installer;

pub use installer::ensure_cuquantum;

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cuda;
use crate::errors::{Error, Result};

/// cuQuantum version we install (major.minor.patch.build)
/// Note: CUDA 13 support requires version 25.09.0.7 or later
pub const CUQUANTUM_VERSION: &str = "25.11.1.11";

/// Get the pecos cuQuantum installation directory
#[must_use]
pub fn get_pecos_cuquantum_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".pecos").join("cuquantum"))
}

/// Find cuQuantum installation, checking local first, then system
///
/// Search order:
/// 1. `~/.pecos/cuquantum/` (local installation)
/// 2. `CUQUANTUM_ROOT` environment variable
/// 3. Standard system paths (`/usr/local/cuquantum`, etc.)
/// 4. Derive from CUDA installation path
#[must_use]
pub fn find_cuquantum() -> Option<PathBuf> {
    // 1. Check ~/.pecos/cuquantum/ first (local installation)
    if let Some(pecos_cuquantum) = get_pecos_cuquantum_dir()
        && is_valid_cuquantum_installation(&pecos_cuquantum)
    {
        return Some(pecos_cuquantum);
    }

    // 2. Check CUQUANTUM_ROOT environment variable
    if let Ok(cuquantum_root) = std::env::var("CUQUANTUM_ROOT") {
        let path = PathBuf::from(&cuquantum_root);
        if is_valid_cuquantum_installation(&path) {
            return Some(path);
        }
    }

    // 3. Check standard system paths
    let standard_paths = [
        "/usr/local/cuquantum",
        "/opt/nvidia/cuquantum",
        "/usr/local/cuda/cuquantum",
    ];

    for path_str in &standard_paths {
        let path = PathBuf::from(path_str);
        if is_valid_cuquantum_installation(&path) {
            return Some(path);
        }
    }

    // 4. Check if cuQuantum is installed alongside CUDA
    if let Some(cuda_path) = cuda::find_cuda() {
        // cuQuantum might be in CUDA's directory structure
        let cuquantum_in_cuda = cuda_path.join("cuquantum");
        if is_valid_cuquantum_installation(&cuquantum_in_cuda) {
            return Some(cuquantum_in_cuda);
        }

        // Or in the same parent directory
        if let Some(parent) = cuda_path.parent() {
            let cuquantum_sibling = parent.join("cuquantum");
            if is_valid_cuquantum_installation(&cuquantum_sibling) {
                return Some(cuquantum_sibling);
            }
        }
    }

    // 5. Try pkg-config
    if let Some(path) = find_via_pkg_config() {
        return Some(path);
    }

    None
}

/// Try to find cuQuantum via pkg-config
fn find_via_pkg_config() -> Option<PathBuf> {
    let output = Command::new("pkg-config")
        .args(["--variable=prefix", "custatevec"])
        .output()
        .ok()?;

    if output.status.success() {
        let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path_str.is_empty() {
            let path = PathBuf::from(&path_str);
            if is_valid_cuquantum_installation(&path) {
                return Some(path);
            }
        }
    }

    None
}

/// Check if a path contains a valid cuQuantum installation
#[must_use]
pub fn is_valid_cuquantum_installation(path: &Path) -> bool {
    if !path.exists() {
        return false;
    }

    // Check for custatevec.h header
    let header = path.join("include").join("custatevec.h");
    if !header.exists() {
        return false;
    }

    // Check for library (try both lib and lib64)
    let lib_names = if cfg!(windows) {
        vec!["custatevec.lib", "custatevec.dll"]
    } else {
        vec!["libcustatevec.so", "libcustatevec.a"]
    };

    let lib_dirs = ["lib", "lib64"];

    for lib_dir in &lib_dirs {
        for lib_name in &lib_names {
            let lib_path = path.join(lib_dir).join(lib_name);
            if lib_path.exists() {
                return true;
            }
        }
    }

    false
}

/// Get cuQuantum version from an installation
///
/// # Errors
/// Returns an error if version cannot be determined.
pub fn get_cuquantum_version(cuquantum_path: &Path) -> Result<String> {
    // Try to read version from header file
    let header = cuquantum_path.join("include").join("custatevec.h");

    if !header.exists() {
        return Err(Error::CuQuantum("custatevec.h not found".into()));
    }

    let content = std::fs::read_to_string(&header)
        .map_err(|e| Error::CuQuantum(format!("Failed to read header: {e}")))?;

    // Look for version define like: #define CUSTATEVEC_VERSION 10200
    // or CUSTATEVEC_VER_MAJOR, CUSTATEVEC_VER_MINOR, CUSTATEVEC_VER_PATCH
    let mut major = None;
    let mut minor = None;
    let mut patch = None;

    for line in content.lines() {
        if line.contains("CUSTATEVEC_VER_MAJOR") {
            major = extract_version_number(line);
        } else if line.contains("CUSTATEVEC_VER_MINOR") {
            minor = extract_version_number(line);
        } else if line.contains("CUSTATEVEC_VER_PATCH") {
            patch = extract_version_number(line);
        }
    }

    match (major, minor, patch) {
        (Some(maj), Some(min), Some(pat)) => Ok(format!("{maj}.{min}.{pat}")),
        (Some(maj), Some(min), None) => Ok(format!("{maj}.{min}.0")),
        _ => {
            // Try alternative version format
            for line in content.lines() {
                if line.contains("CUSTATEVEC_VERSION")
                    && !line.contains("VER_")
                    && let Some(ver) = extract_version_number(line)
                {
                    // Version might be encoded as 10200 for 1.2.0
                    let major = ver / 10000;
                    let minor = (ver % 10000) / 100;
                    let patch = ver % 100;
                    return Ok(format!("{major}.{minor}.{patch}"));
                }
            }
            Err(Error::CuQuantum(
                "Could not parse cuQuantum version from header".into(),
            ))
        }
    }
}

/// Extract version number from a #define line
fn extract_version_number(line: &str) -> Option<u32> {
    line.split_whitespace()
        .last()
        .and_then(|s| s.parse::<u32>().ok())
}

/// Check if cuQuantum is available (either local or system)
#[must_use]
pub fn is_cuquantum_available() -> bool {
    find_cuquantum().is_some()
}

/// Get the library directory within a cuQuantum installation
#[must_use]
pub fn get_lib_dir(cuquantum_path: &Path) -> Option<PathBuf> {
    // Try lib64 first (common on Linux x86_64)
    let lib64 = cuquantum_path.join("lib64");
    if lib64.exists() {
        return Some(lib64);
    }

    let lib = cuquantum_path.join("lib");
    if lib.exists() {
        return Some(lib);
    }

    None
}

/// Get the include directory within a cuQuantum installation
#[must_use]
pub fn get_include_dir(cuquantum_path: &Path) -> PathBuf {
    cuquantum_path.join("include")
}

/// cuQuantum component libraries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CuQuantumLib {
    /// cuStateVec - state vector simulation
    StateVec,
    /// cuTensorNet - tensor network contraction
    TensorNet,
    /// cuDensityMat - density matrix simulation
    DensityMat,
}

impl CuQuantumLib {
    /// Get the library name (without prefix/suffix)
    #[must_use]
    pub fn lib_name(self) -> &'static str {
        match self {
            Self::StateVec => "custatevec",
            Self::TensorNet => "cutensornet",
            Self::DensityMat => "cudensitymat",
        }
    }

    /// Get the header file name
    #[must_use]
    pub fn header_name(self) -> &'static str {
        match self {
            Self::StateVec => "custatevec.h",
            Self::TensorNet => "cutensornet.h",
            Self::DensityMat => "cudensitymat.h",
        }
    }

    /// Check if this library is available in the installation
    #[must_use]
    pub fn is_available(self, cuquantum_path: &Path) -> bool {
        let header = cuquantum_path.join("include").join(self.header_name());
        header.exists()
    }
}

/// Information about a cuQuantum installation
#[derive(Debug)]
pub struct CuQuantumInfo {
    /// Path to cuQuantum installation
    pub path: PathBuf,
    /// Version string
    pub version: Option<String>,
    /// Available libraries
    pub available_libs: Vec<CuQuantumLib>,
    /// Whether CUDA is also available
    pub cuda_available: bool,
    /// CUDA path if available
    pub cuda_path: Option<PathBuf>,
}

/// Get comprehensive information about a cuQuantum installation
///
/// # Errors
/// Returns an error if cuQuantum is not found.
pub fn get_cuquantum_info() -> Result<CuQuantumInfo> {
    let path = find_cuquantum().ok_or_else(|| Error::CuQuantum("cuQuantum not found".into()))?;

    let version = get_cuquantum_version(&path).ok();

    let available_libs = [
        CuQuantumLib::StateVec,
        CuQuantumLib::TensorNet,
        CuQuantumLib::DensityMat,
    ]
    .into_iter()
    .filter(|lib| lib.is_available(&path))
    .collect();

    let cuda_path = cuda::find_cuda();
    let cuda_available = cuda_path.is_some();

    Ok(CuQuantumInfo {
        path,
        version,
        available_libs,
        cuda_available,
        cuda_path,
    })
}

/// Print cargo build script directives for linking cuQuantum
///
/// Call this from your build.rs to set up linking.
///
/// # Errors
/// Returns an error if cuQuantum is not found.
pub fn print_cargo_link_directives(libs: &[CuQuantumLib]) -> Result<()> {
    let cuquantum_path =
        find_cuquantum().ok_or_else(|| Error::CuQuantum("cuQuantum not found".into()))?;

    let lib_dir = get_lib_dir(&cuquantum_path)
        .ok_or_else(|| Error::CuQuantum("cuQuantum lib directory not found".into()))?;

    let include_dir = get_include_dir(&cuquantum_path);

    // Print search path
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Print library links
    for lib in libs {
        println!("cargo:rustc-link-lib={}", lib.lib_name());
    }

    // Print include path for bindgen
    println!("cargo:include={}", include_dir.display());

    // Rerun if cuQuantum installation changes
    println!("cargo:rerun-if-env-changed=CUQUANTUM_ROOT");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lib_names() {
        assert_eq!(CuQuantumLib::StateVec.lib_name(), "custatevec");
        assert_eq!(CuQuantumLib::TensorNet.lib_name(), "cutensornet");
        assert_eq!(CuQuantumLib::DensityMat.lib_name(), "cudensitymat");
    }

    #[test]
    fn test_header_names() {
        assert_eq!(CuQuantumLib::StateVec.header_name(), "custatevec.h");
        assert_eq!(CuQuantumLib::TensorNet.header_name(), "cutensornet.h");
        assert_eq!(CuQuantumLib::DensityMat.header_name(), "cudensitymat.h");
    }

    #[test]
    fn test_extract_version_number() {
        assert_eq!(
            extract_version_number("#define CUSTATEVEC_VER_MAJOR 1"),
            Some(1)
        );
        assert_eq!(
            extract_version_number("#define CUSTATEVEC_VER_MINOR 2"),
            Some(2)
        );
        assert_eq!(
            extract_version_number("#define CUSTATEVEC_VERSION 10200"),
            Some(10200)
        );
        assert_eq!(extract_version_number("// comment"), None);
    }

    #[test]
    fn test_is_cuquantum_available() {
        // This test just verifies the function runs without panic
        // Actual availability depends on the system
        let _ = is_cuquantum_available();
    }

    #[test]
    fn test_find_cuquantum_returns_valid_or_none() {
        if let Some(path) = find_cuquantum() {
            assert!(
                is_valid_cuquantum_installation(&path),
                "find_cuquantum returned invalid path"
            );
        }
    }
}
