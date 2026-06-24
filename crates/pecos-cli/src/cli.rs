//! PECOS CLI — dependency management, CUDA-aware builds, and system inspection.
//!
//! The CLI owns things that need real program logic: detecting CUDA/LLVM/GPU,
//! installing dependencies, introspecting cargo features, and building with
//! platform-specific flags (e.g., macOS rpath handling in `python build`).
//!
//! Daily dev workflows (fmt, test, lint, bench, docs) live in the Justfile,
//! which is transparent, editable, and doesn't require compiling anything.

#![allow(clippy::fn_params_excessive_bools)]

pub mod cuda_cmd;
pub mod cuquantum_cmd;
pub mod env_cmd;
pub mod gpu_cmd;
pub mod info;
pub mod install_cmd;
pub mod list;
pub mod llvm_cmd;
pub mod manifest_cmd;
pub mod migrate_cmd;
pub mod python_cmd;
pub mod rust_cmd;
pub mod selene_cmd;
pub mod setup_cmd;
pub mod uninstall_cmd;
pub mod upgrade_cmd;

use clap::{Subcommand, ValueEnum};

#[derive(Subcommand, Clone)]
pub enum RustCommands {
    /// Run cargo check with CUDA-aware feature handling
    Check {
        /// Also check FFI crates (pecos-rslib, pecos-rslib-cuda, pecos-julia-ffi,
        /// pecos-go-ffi). pecos-rslib-cuda transitively pulls in pecos-cuquantum,
        /// whose Linux build script may download cuTensor over the network if it
        /// isn't already cached in ~/.pecos/deps/; pecos-julia-ffi and pecos-go-ffi
        /// also need Julia/Go installed.
        #[arg(long)]
        include_ffi: bool,
    },

    /// Run cargo clippy with CUDA-aware feature handling
    Clippy {
        /// Also clippy FFI crates (pecos-rslib, pecos-rslib-cuda, pecos-julia-ffi,
        /// pecos-go-ffi). Same external-toolchain caveats as `rust check
        /// --include-ffi`.
        #[arg(long)]
        include_ffi: bool,

        /// Apply clippy fixes (--fix --allow-staged --allow-dirty)
        #[arg(long)]
        fix: bool,
    },

    /// Run cargo test with CUDA-aware feature handling
    Test {
        /// Build profile for tests (dev/debug, release, native)
        #[arg(long, value_enum, default_value = "dev")]
        profile: BuildProfile,

        /// Also test FFI crates (pecos-rslib, pecos-rslib-cuda, pecos-julia-ffi,
        /// pecos-go-ffi). Same external-toolchain caveats as `rust check
        /// --include-ffi`.
        #[arg(long)]
        include_ffi: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BuildProfile {
    Dev,
    Debug,
    Release,
    Native,
}

impl BuildProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dev => "dev",
            Self::Debug => "debug",
            Self::Release => "release",
            Self::Native => "native",
        }
    }
}

#[derive(Subcommand, Clone)]
pub enum PythonCommands {
    /// Build pecos-rslib and quantum-pecos via maturin
    Build {
        /// Build profile (dev/debug, release, native)
        #[arg(long, value_enum, default_value = "dev")]
        profile: BuildProfile,

        /// Additional RUSTFLAGS
        #[arg(long)]
        rustflags: Option<String>,

        /// Force CUDA support on (overrides auto-detection)
        #[arg(long, conflicts_with = "no_cuda")]
        cuda: bool,

        /// Force CUDA support off (overrides auto-detection)
        #[arg(long = "no-cuda")]
        no_cuda: bool,
    },
}

#[derive(Subcommand, Clone)]
pub enum CudaCommands {
    /// Check if CUDA is available
    Check {
        #[arg(short, long)]
        quiet: bool,
    },
    /// Find CUDA installation path
    Find {
        #[arg(long)]
        export: bool,
    },
    /// Show CUDA version
    Version,
    /// Validate CUDA installation
    Validate { path: Option<String> },
    /// Install CUDA Python packages (cupy, cuquantum, pytket-cutensornet)
    SetupPython,
}

#[derive(Subcommand, Clone)]
pub enum CuQuantumCommands {
    /// Check if cuQuantum is available
    Check {
        #[arg(short, long)]
        quiet: bool,
    },
    /// Find cuQuantum installation path
    Find {
        #[arg(long)]
        export: bool,
    },
    /// Show cuQuantum version
    Version,
    /// Validate cuQuantum installation
    Validate { path: Option<String> },
    /// Configure .cargo/config.toml with cuQuantum path
    Configure,
}

#[derive(Subcommand, Clone)]
pub enum GpuCommands {
    /// Check if a GPU (wgpu adapter) is available
    Check {
        #[arg(short, long)]
        quiet: bool,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Clone)]
pub enum SeleneCommands {
    /// Install Selene plugins by copying built libraries to Python packages
    Install {
        #[arg(short, long)]
        plugin: Option<String>,
        #[arg(long, default_value = "dev")]
        profile: String,
        #[arg(long)]
        dry_run: bool,
    },
    /// Clean Selene plugin _dist directories and venv installations
    Clean {
        #[arg(short, long)]
        plugin: Option<String>,
        #[arg(long)]
        venv: bool,
        #[arg(long)]
        dry_run: bool,
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },
    /// List Selene plugins and their installation status
    List,
}

#[derive(Subcommand, Clone)]
pub enum LlvmCommands {
    /// Check if LLVM 21.1 is available
    Check {
        #[arg(short, long)]
        quiet: bool,
    },
    /// Ensure LLVM 21.1 is installed and runtime-valid
    Ensure {
        /// Require the PECOS-managed installation under ~/.pecos/deps
        #[arg(long)]
        managed: bool,

        /// Skip automatic .cargo/config.toml configuration
        #[arg(long)]
        no_configure: bool,
    },
    /// Configure .cargo/config.toml with detected LLVM or an explicit LLVM path
    Configure {
        /// LLVM installation prefix to configure, e.g. /usr/lib/llvm-21
        path: Option<String>,
    },
    /// Find LLVM installation path
    Find {
        #[arg(long)]
        export: bool,
    },
    /// Show LLVM version
    Version,
    /// Validate LLVM installation
    Validate { path: Option<String> },
    /// Find a specific LLVM tool (e.g., llvm-as, clang)
    Tool { name: String },
}

#[derive(Subcommand, Clone)]
pub enum DepsCommands {
    /// Check consistency of shared dependencies across per-crate pecos.toml files
    Check,
    /// Show current manifest status
    Status,
    /// Verify dependency checksums
    Verify {
        #[arg(short, long)]
        deps: Option<String>,
    },
    /// List available dependencies (merged from all per-crate manifests)
    List,
}
