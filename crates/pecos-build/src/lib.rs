//! PECOS build utilities and dependency management
//!
//! This crate provides build script utilities for managing external dependencies:
//!
//! - Downloading and extracting C++ libraries (`QuEST`, Qulacs, Stim, etc.)
//! - Managing LLVM 14 installation
//! - Managing the `~/.pecos/` home directory
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
//! ```no_run
//! use pecos_build::{Manifest, ensure_dep_ready};
//!
//! // Load manifest
//! let manifest = Manifest::find_and_load_validated()
//!     .expect("pecos.toml not found");
//!
//! // Ensure dependency is downloaded and extracted to ~/.pecos/deps/
//! // This persists across `cargo clean` for faster rebuilds
//! let qulacs_path = ensure_dep_ready("qulacs", &manifest)
//!     .expect("Failed to get qulacs");
//! let eigen_path = ensure_dep_ready("eigen", &manifest)
//!     .expect("Failed to get eigen");
//!
//! // Use the paths in your build (example with cc::Build)
//! // build.include(&qulacs_path.join("src"));
//! // build.include(&eigen_path);
//! println!("qulacs: {}", qulacs_path.display());
//! println!("eigen: {}", eigen_path.display());
//! ```
//!
//! Each published crate includes its own `pecos.toml` with the dependencies it needs,
//! so crates.io users automatically get the correct versions.

pub mod cuda;
pub mod cuquantum;
pub mod cutensor;
pub mod deps;
pub mod download;
pub mod errors;
pub mod extract;
pub mod home;
pub mod llvm;
pub mod manifest;
pub mod prompt;

// Re-export main types for convenience
pub use deps::ensure_dep_ready;
pub use download::{DownloadInfo, download_all_cached, download_cached};
pub use errors::{Error, Result};
pub use extract::{extract_archive, extract_to_deps};
pub use home::{
    get_cache_dir, get_cuda_dir, get_cuquantum_dir, get_deps_dir, get_llvm_dir, get_pecos_home,
    get_tmp_dir,
};
pub use manifest::Manifest;

/// Check that the C++ toolchain supports C++20 with the CXX crate.
///
/// On macOS, Xcode 15.x ships a libc++ (based on LLVM 16) that has not implemented
/// LWG 3545 (SFINAE-friendly `std::pointer_traits`). The CXX crate's
/// `rust::Slice<T>::iterator` uses `contiguous_iterator_tag` in C++20 mode, which
/// triggers `pointer_traits` instantiation that fails on these older libc++ versions.
/// Xcode 16+ (LLVM 18) has the fix.
///
/// Call this from build scripts that use `cxx_build` with C++20 to give users a
/// clear error instead of a cryptic template instantiation failure.
///
/// # Panics
///
/// Panics with a descriptive message if the toolchain is known to be incompatible.
pub fn check_cxx20_toolchain() {
    // Only relevant on macOS with Apple Clang
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    // Determine which C++ compiler will be used
    let compiler = std::env::var("CXX")
        .or_else(|_| std::env::var("CC"))
        .unwrap_or_else(|_| "clang".to_string());

    let Ok(output) = std::process::Command::new(&compiler)
        .args(["-dM", "-E", "-x", "c++", "/dev/null"])
        .output()
    else {
        return; // Can't check, let compilation proceed
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(version_str) = line
            .strip_prefix("#define __apple_build_version__ ")
            .map(str::trim)
        {
            if let Ok(version) = version_str.parse::<u64>() {
                assert!(
                    version >= 16_000_000,
                    "\n\n\
                    Xcode 16 or later is required for C++20 builds with the CXX crate.\n\
                    \n\
                    Detected Apple Clang build version: {version}\n\
                    \n\
                    Xcode 15.x ships a libc++ that has not implemented LWG 3545\n\
                    (SFINAE-friendly std::pointer_traits), which causes compilation\n\
                    failures with CXX crate iterators in C++20 mode.\n\
                    \n\
                    To fix: upgrade to Xcode 16+ or macOS 15+.\n\
                    See: https://github.com/dtolnay/cxx/issues/1436\n\n"
                );
            }
            break;
        }
    }
}

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
