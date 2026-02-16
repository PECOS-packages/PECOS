//! CLI command definitions and handlers for PECOS developer tools
//!
//! This module contains the command definitions and implementations for all
//! dev tool commands. The command enums are designed to be embedded in the
//! main pecos CLI.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::fn_params_excessive_bools)]

pub mod cuda_cmd;
pub mod docs_cmd;
pub mod features_cmd;
pub mod go_cmd;
pub mod info;
pub mod julia_cmd;
pub mod list;
pub mod llvm_cmd;
pub mod manifest_cmd;
pub mod python_cmd;
pub mod rust_cmd;
pub mod selene_cmd;

use clap::Subcommand;

// ============================================================================
// Rust Commands
// ============================================================================

#[derive(Subcommand, Clone)]
pub enum RustCommands {
    /// Run cargo check with CUDA-aware feature handling
    ///
    /// If CUDA is not available, automatically excludes GPU features from
    /// pecos and pecos-quest packages.
    Check {
        /// Also check FFI crates (pecos-rslib, pecos-julia-ffi, pecos-go-ffi)
        #[arg(long)]
        include_ffi: bool,
    },

    /// Run cargo clippy with CUDA-aware feature handling
    Clippy {
        /// Also check FFI crates (pecos-rslib, pecos-julia-ffi, pecos-go-ffi)
        #[arg(long)]
        include_ffi: bool,

        /// Apply clippy fixes (--fix --allow-staged --allow-dirty)
        #[arg(long)]
        fix: bool,
    },

    /// Run cargo test with CUDA-aware feature handling
    Test {
        /// Use release mode for tests
        #[arg(long)]
        release: bool,

        /// Also test FFI crates
        #[arg(long)]
        include_ffi: bool,
    },

    /// Run cargo fmt
    Fmt {
        /// Check formatting without modifying files
        #[arg(long)]
        check: bool,
    },
}

// ============================================================================
// Python Commands
// ============================================================================

#[derive(Subcommand, Clone)]
pub enum PythonCommands {
    /// Check if Python/uv is available
    Check {
        /// Suppress output (exit code only)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Build pecos-rslib and quantum-pecos
    ///
    /// Uses maturin to build the Rust library and installs quantum-pecos
    /// in editable mode.
    Build {
        /// Build profile (dev/debug, release, native)
        #[arg(long, default_value = "dev")]
        profile: String,

        /// Additional RUSTFLAGS (e.g., "-C target-cpu=native")
        #[arg(long)]
        rustflags: Option<String>,

        /// Build with CUDA support
        #[arg(long)]
        cuda: bool,
    },

    /// Run Python tests with pytest
    Test {
        /// Pytest markers to filter tests (e.g., "not slow")
        #[arg(short, long)]
        markers: Option<String>,

        /// Increase verbosity (-v, -vv)
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,

        /// Run Selene plugin tests instead of core tests
        #[arg(long)]
        selene: bool,

        /// Run NumPy/SciPy compatibility tests
        #[arg(long)]
        numpy: bool,
    },
}

// ============================================================================
// CUDA Commands
// ============================================================================

#[derive(Subcommand, Clone)]
pub enum CudaCommands {
    /// Download and install CUDA Toolkit to ~/.pecos/cuda/
    Install {
        /// Force reinstall even if already present
        #[arg(long)]
        force: bool,
    },

    /// Check if CUDA is available (local or system)
    Check {
        /// Suppress output (exit code only)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Find CUDA installation path
    Find {
        /// Print export command for shell evaluation
        #[arg(long)]
        export: bool,
    },

    /// Show CUDA version information
    Version,

    /// Remove local CUDA installation (~/.pecos/cuda/)
    Uninstall,

    /// Validate CUDA installation integrity
    Validate {
        /// Path to CUDA installation (uses detected path if not specified)
        path: Option<String>,
    },

    /// Install CUDA Python packages (cupy, cuquantum, pytket-cutensornet)
    ///
    /// Requires CUDA toolkit to be installed first (pecos cuda install or system CUDA).
    /// Installs quantum-pecos[cuda] which includes cupy, cuquantum, and pytket-cutensornet.
    SetupPython,
}

// ============================================================================
// Julia Commands
// ============================================================================

#[derive(Subcommand, Clone)]
pub enum JuliaCommands {
    /// Check if Julia is available
    Check {
        /// Suppress output (exit code only)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Build Julia FFI library
    Build {
        /// Build profile (dev/debug, release, native)
        #[arg(long, default_value = "dev")]
        profile: String,

        /// Additional RUSTFLAGS (e.g., "-C target-cpu=native")
        #[arg(long)]
        rustflags: Option<String>,
    },

    /// Run Julia tests
    Test,

    /// Format Julia code
    Fmt {
        /// Check formatting without modifying files
        #[arg(long)]
        check: bool,
    },

    /// Run Julia linting (Aqua.jl)
    Lint,
}

// ============================================================================
// Go Commands
// ============================================================================

#[derive(Subcommand, Clone)]
pub enum GoCommands {
    /// Check if Go is available
    Check {
        /// Suppress output (exit code only)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Build Go FFI library
    Build {
        /// Build profile (dev/debug, release, native)
        #[arg(long, default_value = "dev")]
        profile: String,

        /// Additional RUSTFLAGS (e.g., "-C target-cpu=native")
        #[arg(long)]
        rustflags: Option<String>,
    },

    /// Run Go tests
    Test,

    /// Format Go code
    Fmt {
        /// Check formatting without modifying files
        #[arg(long)]
        check: bool,
    },

    /// Run Go linting (go vet)
    Lint,
}

// ============================================================================
// Selene Commands
// ============================================================================

#[derive(Subcommand, Clone)]
pub enum SeleneCommands {
    /// Install Selene plugins by copying built libraries to Python packages
    Install {
        /// Specific plugin to install (default: all)
        #[arg(short, long)]
        plugin: Option<String>,

        /// Build profile to use (dev/debug, release, native)
        #[arg(long, default_value = "dev")]
        profile: String,

        /// Show what would be copied without copying
        #[arg(long)]
        dry_run: bool,
    },

    /// Clean Selene plugin _dist directories and venv installations
    Clean {
        /// Specific plugin to clean (default: all)
        #[arg(short, long)]
        plugin: Option<String>,

        /// Also clean plugins from .venv/lib/*/site-packages/
        #[arg(long)]
        venv: bool,

        /// Show what would be deleted without deleting
        #[arg(long)]
        dry_run: bool,

        /// Increase verbosity (-v, -vv, -vvv)
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },

    /// List Selene plugins and their installation status
    List,
}

// ============================================================================
// Features Commands
// ============================================================================

#[derive(Subcommand, Clone)]
pub enum FeaturesCommands {
    /// List features for a package
    List {
        /// Package name (e.g., pecos, pecos-quest)
        #[arg(short, long)]
        package: String,

        /// Features to exclude (comma-separated, e.g., "gpu,cuda")
        #[arg(short, long)]
        exclude: Option<String>,

        /// Output as JSON array
        #[arg(long)]
        json: bool,
    },
}

// ============================================================================
// LLVM Commands
// ============================================================================

#[derive(Subcommand, Clone)]
pub enum LlvmCommands {
    /// Download and install LLVM 14
    Install {
        /// Force reinstall even if already present
        #[arg(long)]
        force: bool,

        /// Skip automatic configuration after installation
        #[arg(long)]
        no_configure: bool,
    },

    /// Check if LLVM 14 is available
    Check {
        /// Suppress output messages
        #[arg(short, long)]
        quiet: bool,
    },

    /// Configure .cargo/config.toml with LLVM path
    Configure,

    /// Find LLVM installation path
    Find {
        /// Print export command for shell evaluation
        #[arg(long)]
        export: bool,
    },

    /// Show LLVM version information
    Version,

    /// Validate LLVM installation integrity
    Validate {
        /// Path to LLVM installation (uses detected path if not specified)
        path: Option<String>,
    },

    /// Find a specific LLVM tool
    Tool {
        /// Name of the tool (e.g., llvm-as, clang)
        name: String,
    },
}

// ============================================================================
// Deps Commands
// ============================================================================

#[derive(Subcommand, Clone)]
pub enum DepsCommands {
    /// Initialize a new pecos.toml manifest
    Init {
        /// Overwrite existing manifest
        #[arg(long)]
        force: bool,
    },

    /// Show current manifest status
    Status,

    /// Sync crate manifests from workspace manifest
    Sync {
        /// Show what would be changed without making changes
        #[arg(long)]
        dry_run: bool,
    },

    /// Verify dependency checksums by downloading and checking
    Verify {
        /// Only verify specific dependencies (comma-separated)
        #[arg(short, long)]
        deps: Option<String>,
    },

    /// List available dependencies
    List,
}

// ============================================================================
// Command Runners
// ============================================================================

/// Run a Rust subcommand
pub fn run_rust(command: &RustCommands) -> pecos_build::Result<()> {
    rust_cmd::run(command)
}

/// Run a Python subcommand
pub fn run_python(command: &PythonCommands) -> pecos_build::Result<()> {
    python_cmd::run(command)
}

/// Run a CUDA subcommand
pub fn run_cuda(command: CudaCommands) -> pecos_build::Result<()> {
    cuda_cmd::run(command)
}

/// Run a Julia subcommand
pub fn run_julia(command: &JuliaCommands) -> pecos_build::Result<()> {
    julia_cmd::run(command)
}

/// Run a Go subcommand
pub fn run_go(command: &GoCommands) -> pecos_build::Result<()> {
    go_cmd::run(command)
}

/// Run a Selene subcommand
pub fn run_selene(command: SeleneCommands) -> pecos_build::Result<()> {
    selene_cmd::run(command)
}

/// Run a Features subcommand
pub fn run_features(command: FeaturesCommands) -> pecos_build::Result<()> {
    features_cmd::run(command)
}

/// Run an LLVM subcommand
pub fn run_llvm(command: LlvmCommands) -> pecos_build::Result<()> {
    llvm_cmd::run(command)
}

/// Run a Deps subcommand
pub fn run_deps(command: DepsCommands) -> pecos_build::Result<()> {
    manifest_cmd::run(command)
}

/// Run the sys-info command
pub fn run_sys_info() -> pecos_build::Result<()> {
    info::run()
}

/// Run the list command
pub fn run_list(verbose: bool) -> pecos_build::Result<()> {
    list::run(verbose)
}

/// Run the docs command
pub fn run_docs(port: u16, no_browser: bool) -> pecos_build::Result<()> {
    docs_cmd::run(port, no_browser)
}
