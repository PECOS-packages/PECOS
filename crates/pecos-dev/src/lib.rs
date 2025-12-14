//! PECOS development tools and dependency management
//!
//! This crate provides:
//!
//! - The `pecos-dev` CLI binary for dependency management
//! - Tools for managing external dependencies (LLVM 14, C++ libraries)
//! - Build script utilities for downloading and extracting dependencies
//!
//! # CLI Usage
//!
//! ```bash
//! pecos-dev llvm install   # Install LLVM 14 to ~/.pecos/llvm/
//! pecos-dev llvm check     # Check LLVM installation status
//! pecos-dev deps sync      # Sync crate manifests from workspace
//! ```
//!
//! # PECOS Home Directory
//!
//! All dependencies are managed under `~/.pecos/`:
//!
//! ```text
//! ~/.pecos/
//! ├── cache/      # Downloaded archives (tar.gz, etc.)
//! ├── deps/       # Extracted source trees (ready for building)
//! ├── llvm/       # LLVM installation
//! └── tmp/        # Temporary files during downloads/extraction
//! ```
//!
//! # Environment Variables
//!
//! - `PECOS_HOME`: Override the entire home directory (default: `~/.pecos/`)
//! - `PECOS_DEPS_DIR`: Override extracted sources location (default: `$PECOS_HOME/deps/`)
//! - `PECOS_CACHE_DIR`: Override archives location (default: `$PECOS_HOME/cache/`)
//! - `RUST_LOG`: Set log level for build output (e.g., `info` for download progress)
//!
//! # Usage in Build Scripts
//!
//! Build scripts should use `ensure_dep_ready()` for dependency management:
//!
//! ```ignore
//! use pecos_dev::{Manifest, ensure_dep_ready};
//!
//! fn main() {
//!     // Load manifest
//!     let manifest = Manifest::find_and_load_validated()
//!         .expect("pecos.toml not found");
//!
//!     // Ensure dependency is downloaded and extracted to ~/.pecos/deps/
//!     // This persists across `cargo clean` for faster rebuilds
//!     let qulacs_path = ensure_dep_ready("qulacs", &manifest)
//!         .expect("Failed to get qulacs");
//!     let eigen_path = ensure_dep_ready("eigen", &manifest)
//!         .expect("Failed to get eigen");
//!
//!     // Use the paths in your build
//!     build.include(&qulacs_path.join("src"));
//!     build.include(&eigen_path);
//! }
//! ```
//!
//! Each published crate includes its own `pecos.toml` with the dependencies it needs,
//! so crates.io users automatically get the correct versions.

pub mod cuda;
pub mod deps;
pub mod download;
pub mod errors;
pub mod extract;
pub mod home;
pub mod llvm;
pub mod manifest;

#[cfg(feature = "cli")]
pub mod cli;

// Re-export main types for convenience
pub use deps::ensure_dep_ready;
pub use download::{DownloadInfo, download_all_cached, download_cached};
pub use errors::{Error, Result};
pub use extract::{extract_archive, extract_to_deps};
pub use home::{get_cache_dir, get_deps_dir, get_llvm_dir, get_pecos_home, get_tmp_dir};
pub use manifest::Manifest;

/// Report ccache/sccache configuration for C++ builds
pub fn report_cache_config() {
    use log::{debug, info};

    info!("Checking C++ compiler cache configuration...");

    let cc = std::env::var("CC").unwrap_or_default();
    let cxx = std::env::var("CXX").unwrap_or_default();

    if cc.contains("ccache") || cc.contains("sccache") {
        info!("Using compiler cache via CC: {cc}");
    } else if cxx.contains("ccache") || cxx.contains("sccache") {
        info!("Using compiler cache via CXX: {cxx}");
    } else if let Ok(wrapper) = std::env::var("RUSTC_WRAPPER") {
        if wrapper.contains("sccache") {
            debug!(
                "Note: RUSTC_WRAPPER=sccache detected. For C++ caching, also set CC='sccache cc' and CXX='sccache c++'"
            );
        } else if wrapper.contains("ccache") {
            debug!(
                "Note: RUSTC_WRAPPER=ccache detected. For C++ caching, also set CC='ccache cc' and CXX='ccache c++'"
            );
        }
    }

    if let Ok(num_jobs) = std::env::var("NUM_JOBS") {
        info!("Using {num_jobs} parallel jobs for C++ compilation");
    }
}
