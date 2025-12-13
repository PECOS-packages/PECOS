//! PECOS command-line interface and dependency management
//!
//! This crate provides:
//!
//! - The `pecos` CLI binary for dependency management and extension discovery
//! - Tools for managing external dependencies (LLVM 14, C++ libraries)
//! - Build script utilities for downloading and extracting dependencies
//!
//! # CLI Usage
//!
//! ```bash
//! pecos llvm install       # Install LLVM 14 to ~/.pecos/llvm/
//! pecos llvm check         # Check LLVM installation status
//! pecos deps sync          # Sync crate manifests from workspace
//! pecos run foo.qir        # Run quantum program (via pecos-run extension)
//! ```
//!
//! # PECOS Home Directory
//!
//! All dependencies are managed under `~/.pecos/`:
//!
//! ```text
//! ~/.pecos/
//! ├── llvm/       # LLVM installations
//! ├── deps/       # Downloaded C++ dependencies
//! ├── cache/      # Build artifacts
//! └── tmp/        # Temporary files during downloads/extraction
//! ```
//!
//! # Environment Variables
//!
//! - `PECOS_HOME`: Override the entire home directory (default: `~/.pecos/`)
//! - `PECOS_DEPS_DIR`: Override deps location (default: `$PECOS_HOME/deps/`)
//! - `PECOS_CACHE_DIR`: Override cache location (default: `$PECOS_HOME/cache/`)
//!
//! # Usage in Build Scripts
//!
//! Build scripts read dependency information from `pecos.toml`:
//!
//! ```ignore
//! use pecos_cli::{Manifest, download_cached, extract_archive};
//!
//! fn main() {
//!     // Load manifest
//!     // Search order:
//!     // 1. CARGO_MANIFEST_DIR/pecos.toml (crate-local, included in published crate)
//!     // 2. Walk up directory tree (workspace-level, for developers)
//!     let manifest = Manifest::find_and_load()
//!         .expect("pecos.toml not found");
//!
//!     // Get download info for a dependency
//!     let info = manifest.get_download_info("quest")
//!         .expect("quest not defined");
//!
//!     // Download (cached) and extract
//!     let data = download_cached(&info).expect("Download failed");
//!     extract_archive(&data, &out_dir, None).expect("Extract failed");
//! }
//! ```
//!
//! Each published crate includes its own `pecos.toml` with the dependencies it needs,
//! so crates.io users automatically get the correct versions.

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
pub use download::{DownloadInfo, download_all_cached, download_cached};
pub use errors::{Error, Result};
pub use extract::extract_archive;
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
