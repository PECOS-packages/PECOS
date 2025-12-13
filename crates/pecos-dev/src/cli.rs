//! CLI implementation for pecos-dev

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::fn_params_excessive_bools)]

mod info;
mod list;
mod llvm_cmd;
mod manifest_cmd;

use clap::{Parser, Subcommand};

/// PECOS developer tools
#[derive(Parser)]
#[command(name = "pecos-dev")]
#[command(about = "PECOS developer tools - LLVM setup, dependency management, and build utilities", long_about = None)]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Show PECOS home directory info and status
    Info,

    /// List installed and cached dependencies
    List {
        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Clean cached dependencies
    Clean {
        /// Clean downloaded dependencies (~/.pecos/deps/)
        #[arg(long)]
        deps: bool,

        /// Clean build cache (~/.pecos/cache/)
        #[arg(long)]
        cache: bool,

        /// Clean everything except LLVM
        #[arg(long)]
        all: bool,

        /// Also remove LLVM installation (requires --all)
        #[arg(long, requires = "all")]
        include_llvm: bool,

        /// Show what would be deleted without deleting
        #[arg(long)]
        dry_run: bool,
    },

    /// LLVM management commands
    Llvm {
        #[command(subcommand)]
        command: LlvmCommands,
    },

    /// Dependency manifest management (pecos.toml)
    Deps {
        #[command(subcommand)]
        command: DepsCommands,
    },
}

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
    ///
    /// Updates each crate's pecos.toml to match the workspace pecos.toml.
    /// Only affects crates listed in [crates.*] that have dependencies.
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

/// Run the CLI
pub fn run() -> crate::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info => info::run(),
        Commands::List { verbose } => list::run(verbose),
        Commands::Clean {
            deps,
            cache,
            all,
            include_llvm,
            dry_run,
        } => run_clean(deps, cache, all, include_llvm, dry_run),
        Commands::Llvm { command } => llvm_cmd::run(command),
        Commands::Deps { command } => manifest_cmd::run(command),
    }
}

fn run_clean(
    deps: bool,
    cache: bool,
    all: bool,
    include_llvm: bool,
    dry_run: bool,
) -> crate::Result<()> {
    use crate::home::{get_cache_dir, get_deps_dir, get_llvm_dir};
    use std::fs;

    let clean_deps = deps || all;
    let clean_cache = cache || all;
    let clean_llvm = include_llvm && all;

    if !clean_deps && !clean_cache && !clean_llvm {
        println!("Nothing to clean. Use --deps, --cache, --all, or --all --include-llvm");
        return Ok(());
    }

    if clean_deps {
        let deps_dir = get_deps_dir()?;
        if deps_dir.exists() {
            if dry_run {
                println!("Would remove: {}", deps_dir.display());
            } else {
                println!("Removing: {}", deps_dir.display());
                fs::remove_dir_all(&deps_dir)?;
            }
        }
    }

    if clean_cache {
        let cache_dir = get_cache_dir()?;
        if cache_dir.exists() {
            if dry_run {
                println!("Would remove: {}", cache_dir.display());
            } else {
                println!("Removing: {}", cache_dir.display());
                fs::remove_dir_all(&cache_dir)?;
            }
        }
    }

    if clean_llvm {
        let llvm_dir = get_llvm_dir()?;
        if llvm_dir.exists() {
            if dry_run {
                println!("Would remove: {}", llvm_dir.display());
            } else {
                println!("Removing: {}", llvm_dir.display());
                fs::remove_dir_all(&llvm_dir)?;
            }
        }
    }

    if dry_run {
        println!();
        println!("(dry run - no files were deleted)");
    } else {
        println!();
        println!("Done.");
    }

    Ok(())
}
