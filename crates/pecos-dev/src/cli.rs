//! CLI implementation for pecos-dev

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::fn_params_excessive_bools)]

mod clean_cmd;
mod cuda_cmd;
mod features_cmd;
mod go_cmd;
mod info;
mod julia_cmd;
mod list;
mod llvm_cmd;
mod manifest_cmd;
mod python_cmd;
mod rust_cmd;
mod selene_cmd;

use clap::{Parser, Subcommand};

/// PECOS developer tools
#[derive(Parser)]
#[command(name = "pecos-dev")]
#[command(about = "PECOS developer tools - build, test, and manage PECOS development", long_about = None)]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Rust/Cargo commands (CUDA-aware)
    #[command(visible_alias = "rs")]
    Rust {
        #[command(subcommand)]
        command: RustCommands,
    },

    /// Python build and test commands
    #[command(visible_alias = "py")]
    Python {
        #[command(subcommand)]
        command: PythonCommands,
    },

    /// CUDA availability and info
    Cuda {
        #[command(subcommand)]
        command: CudaCommands,
    },

    /// Julia build and test commands
    #[command(visible_alias = "jl")]
    Julia {
        #[command(subcommand)]
        command: JuliaCommands,
    },

    /// Go build and test commands
    Go {
        #[command(subcommand)]
        command: GoCommands,
    },

    /// LLVM 14 management
    Llvm {
        #[command(subcommand)]
        command: LlvmCommands,
    },

    /// Selene plugin management
    Selene {
        #[command(subcommand)]
        command: SeleneCommands,
    },

    /// Clean build artifacts and caches
    Clean {
        #[command(subcommand)]
        command: CleanCommands,
    },

    /// Query package features
    Features {
        #[command(subcommand)]
        command: FeaturesCommands,
    },

    /// Dependency manifest management (pecos.toml)
    Deps {
        #[command(subcommand)]
        command: DepsCommands,
    },

    /// Show system tools and project info
    #[command(name = "sys-info")]
    SysInfo,

    /// List installed and cached dependencies
    List {
        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },
}

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
        /// Build profile (debug, release, native)
        #[arg(long, default_value = "debug")]
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
}

// ============================================================================
// Julia Commands
// ============================================================================

#[derive(Subcommand)]
pub enum JuliaCommands {
    /// Check if Julia is available
    Check {
        /// Suppress output (exit code only)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Build Julia FFI library
    Build {
        /// Build profile (debug, release, native)
        #[arg(long, default_value = "release")]
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

#[derive(Subcommand)]
pub enum GoCommands {
    /// Check if Go is available
    Check {
        /// Suppress output (exit code only)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Build Go FFI library
    Build {
        /// Build profile (debug, release, native)
        #[arg(long, default_value = "release")]
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
// LLVM Commands
// ============================================================================

#[derive(Subcommand)]
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
// Selene Commands
// ============================================================================

#[derive(Subcommand)]
pub enum SeleneCommands {
    /// Install Selene plugins by copying built libraries to Python packages
    Install {
        /// Specific plugin to install (default: all)
        #[arg(short, long)]
        plugin: Option<String>,

        /// Build profile to use (debug, release, native)
        #[arg(long, default_value = "release")]
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
// Clean Commands
// ============================================================================

#[derive(Subcommand, Clone, Copy)]
pub enum CleanCommands {
    /// Clean build artifacts (Python, Rust, Julia)
    ///
    /// Removes Python build artifacts, test caches, compiled extensions,
    /// and optionally runs cargo clean.
    Build {
        /// Show what would be deleted without deleting
        #[arg(long)]
        dry_run: bool,

        /// Skip running cargo clean
        #[arg(long)]
        skip_cargo: bool,

        /// Increase verbosity (-v, -vv, -vvv)
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },

    /// Clean ~/.pecos/deps/ (extracted C++ dependencies)
    Deps {
        /// Increase verbosity (-v, -vv, -vvv)
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },

    /// Clean ~/.pecos/cache/ and tmp/ (downloaded archives)
    Cache {
        /// Increase verbosity (-v, -vv, -vvv)
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },

    /// Clean ~/.pecos/llvm/ (LLVM installation)
    Llvm {
        /// Increase verbosity (-v, -vv, -vvv)
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },

    /// Clean ~/.pecos/cuda/ (CUDA installation)
    Cuda {
        /// Increase verbosity (-v, -vv, -vvv)
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },

    /// Clean deps, cache, and tmp (optionally including LLVM and CUDA)
    All {
        /// Also remove LLVM installation
        #[arg(long)]
        include_llvm: bool,

        /// Also remove CUDA installation
        #[arg(long)]
        include_cuda: bool,

        /// Increase verbosity (-v, -vv, -vvv)
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },
}

// ============================================================================
// Features Commands
// ============================================================================

#[derive(Subcommand)]
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
// Deps Commands
// ============================================================================

#[derive(Subcommand)]
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
}

// ============================================================================
// CLI Runner
// ============================================================================

/// Run the CLI
pub fn run() -> crate::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Rust { command } => rust_cmd::run(&command),
        Commands::Python { command } => python_cmd::run(&command),
        Commands::Cuda { command } => cuda_cmd::run(command),
        Commands::Julia { command } => julia_cmd::run(&command),
        Commands::Go { command } => go_cmd::run(&command),
        Commands::Llvm { command } => llvm_cmd::run(command),
        Commands::Selene { command } => selene_cmd::run(command),
        Commands::Clean { command } => clean_cmd::run(command),
        Commands::Features { command } => features_cmd::run(command),
        Commands::Deps { command } => manifest_cmd::run(command),
        Commands::SysInfo => info::run(),
        Commands::List { verbose } => list::run(verbose),
    }
}
