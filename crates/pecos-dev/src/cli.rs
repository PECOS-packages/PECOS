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

    /// Clean cached dependencies and build artifacts
    Clean {
        /// Clean extracted source trees (~/.pecos/deps/)
        #[arg(long)]
        deps: bool,

        /// Clean downloaded archives (~/.pecos/cache/) and tmp/
        #[arg(long)]
        cache: bool,

        /// Clean LLVM installation (~/.pecos/llvm/)
        #[arg(long)]
        llvm: bool,

        /// Clean deps, cache, and tmp (but not LLVM)
        #[arg(long)]
        all: bool,

        /// Also remove LLVM when using --all (shortcut for --all --llvm)
        #[arg(long)]
        include_llvm: bool,

        /// Clean a specific dependency by name (from deps/ and cache/)
        #[arg(long, value_name = "NAME")]
        dep: Option<String>,

        /// Clean stale archives misplaced in deps/ (from before restructuring)
        #[arg(long)]
        stale: bool,

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
            llvm,
            all,
            include_llvm,
            dep,
            stale,
            dry_run,
        } => run_clean(&CleanOptions {
            deps,
            cache,
            llvm,
            all,
            include_llvm,
            dep,
            stale,
            dry_run,
        }),
        Commands::Llvm { command } => llvm_cmd::run(command),
        Commands::Deps { command } => manifest_cmd::run(command),
    }
}

/// Options for the clean command
#[allow(clippy::struct_excessive_bools)]
struct CleanOptions {
    deps: bool,
    cache: bool,
    llvm: bool,
    all: bool,
    include_llvm: bool,
    dep: Option<String>,
    stale: bool,
    dry_run: bool,
}

/// Get the size of a directory recursively
fn get_dir_size(path: &std::path::Path) -> u64 {
    if !path.exists() {
        return 0;
    }

    let mut size = 0;
    if path.is_file() {
        return path.metadata().map(|m| m.len()).unwrap_or(0);
    }

    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                size += get_dir_size(&entry_path);
            } else {
                size += entry.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    size
}

/// Format bytes into human-readable size
#[allow(clippy::cast_precision_loss)]
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} bytes")
    }
}

#[allow(clippy::too_many_lines)]
fn run_clean(opts: &CleanOptions) -> crate::Result<()> {
    use crate::home::{get_cache_dir, get_deps_dir, get_llvm_dir, get_tmp_dir};
    use std::fs;

    let clean_deps = opts.deps || opts.all;
    let clean_cache = opts.cache || opts.all;
    let clean_llvm = opts.llvm || opts.include_llvm;
    let clean_specific_dep = opts.dep.is_some();
    let clean_stale = opts.stale;

    // Check if anything to do
    if !clean_deps && !clean_cache && !clean_llvm && !clean_specific_dep && !clean_stale {
        println!("Nothing to clean. Options:");
        println!("  --deps        Clean extracted sources (~/.pecos/deps/)");
        println!("  --cache       Clean downloaded archives (~/.pecos/cache/)");
        println!("  --llvm        Clean LLVM installation (~/.pecos/llvm/)");
        println!("  --all         Clean deps + cache + tmp");
        println!("  --dep <NAME>  Clean a specific dependency");
        println!("  --stale       Clean stale archives in deps/");
        return Ok(());
    }

    let mut total_freed: u64 = 0;
    let dry_run = opts.dry_run;

    // Clean specific dependency
    if let Some(ref dep_name) = opts.dep {
        total_freed += clean_specific_dependency(dep_name, dry_run)?;
    }

    // Clean stale archives in deps/
    if clean_stale {
        total_freed += clean_stale_archives(dry_run)?;
    }

    // Clean deps directory
    if clean_deps {
        let deps_dir = get_deps_dir()?;
        if deps_dir.exists() {
            let size = get_dir_size(&deps_dir);
            if dry_run {
                println!(
                    "Would remove: {} ({})",
                    deps_dir.display(),
                    format_size(size)
                );
            } else {
                println!("Removing: {} ({})", deps_dir.display(), format_size(size));
                fs::remove_dir_all(&deps_dir)?;
            }
            total_freed += size;
        }
    }

    // Clean cache directory
    if clean_cache {
        let cache_dir = get_cache_dir()?;
        if cache_dir.exists() {
            let size = get_dir_size(&cache_dir);
            if dry_run {
                println!(
                    "Would remove: {} ({})",
                    cache_dir.display(),
                    format_size(size)
                );
            } else {
                println!("Removing: {} ({})", cache_dir.display(), format_size(size));
                fs::remove_dir_all(&cache_dir)?;
            }
            total_freed += size;
        }

        // Also clean tmp/
        let tmp_dir = get_tmp_dir()?;
        if tmp_dir.exists() {
            let size = get_dir_size(&tmp_dir);
            if dry_run {
                println!(
                    "Would remove: {} ({})",
                    tmp_dir.display(),
                    format_size(size)
                );
            } else {
                println!("Removing: {} ({})", tmp_dir.display(), format_size(size));
                fs::remove_dir_all(&tmp_dir)?;
            }
            total_freed += size;
        }
    }

    // Clean LLVM directory
    if clean_llvm {
        let llvm_dir = get_llvm_dir()?;
        if llvm_dir.exists() {
            let size = get_dir_size(&llvm_dir);
            if dry_run {
                println!(
                    "Would remove: {} ({})",
                    llvm_dir.display(),
                    format_size(size)
                );
            } else {
                println!("Removing: {} ({})", llvm_dir.display(), format_size(size));
                fs::remove_dir_all(&llvm_dir)?;
            }
            total_freed += size;
        }
    }

    // Summary
    println!();
    if total_freed > 0 {
        if dry_run {
            println!("Total: {} would be freed", format_size(total_freed));
            println!("(dry run - no files were deleted)");
        } else {
            println!("Done. Freed {}.", format_size(total_freed));
        }
    } else {
        println!("Nothing to clean.");
    }

    Ok(())
}

/// Clean a specific dependency from deps/ and cache/
#[allow(clippy::collapsible_if)]
fn clean_specific_dependency(dep_name: &str, dry_run: bool) -> crate::Result<u64> {
    use crate::home::{get_cache_dir, get_deps_dir};
    use std::fs;

    let mut total_freed: u64 = 0;
    let deps_dir = get_deps_dir()?;
    let cache_dir = get_cache_dir()?;

    // Find matching directories in deps/
    if deps_dir.exists() {
        if let Ok(entries) = fs::read_dir(&deps_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(dep_name) && entry.path().is_dir() {
                    let size = get_dir_size(&entry.path());
                    if dry_run {
                        println!(
                            "Would remove: {} ({})",
                            entry.path().display(),
                            format_size(size)
                        );
                    } else {
                        println!(
                            "Removing: {} ({})",
                            entry.path().display(),
                            format_size(size)
                        );
                        fs::remove_dir_all(entry.path())?;
                    }
                    total_freed += size;
                }
            }
        }
    }

    // Find matching archives in cache/
    if cache_dir.exists() {
        if let Ok(entries) = fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(dep_name) && entry.path().is_file() {
                    let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    if dry_run {
                        println!(
                            "Would remove: {} ({})",
                            entry.path().display(),
                            format_size(size)
                        );
                    } else {
                        println!(
                            "Removing: {} ({})",
                            entry.path().display(),
                            format_size(size)
                        );
                        fs::remove_file(entry.path())?;
                    }
                    total_freed += size;
                }
            }
        }
    }

    if total_freed == 0 {
        println!("No files found matching '{dep_name}'");
    }

    Ok(total_freed)
}

/// Clean stale .tar.gz archives that are in deps/ instead of cache/
#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn clean_stale_archives(dry_run: bool) -> crate::Result<u64> {
    use crate::home::get_deps_dir;
    use std::fs;

    let mut total_freed: u64 = 0;
    let deps_dir = get_deps_dir()?;

    if !deps_dir.exists() {
        return Ok(0);
    }

    let mut found_stale = false;
    if let Ok(entries) = fs::read_dir(&deps_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            // Look for archive files that shouldn't be in deps/
            if entry.path().is_file()
                && (name.ends_with(".tar.gz")
                    || name.ends_with(".tar.bz2")
                    || name.ends_with(".tar.xz")
                    || name.ends_with(".7z")
                    || name.ends_with(".zip"))
            {
                if !found_stale {
                    println!("Found stale archives in deps/ (should be in cache/):");
                    found_stale = true;
                }

                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                if dry_run {
                    println!("  Would remove: {} ({})", name, format_size(size));
                } else {
                    println!("  Removing: {} ({})", name, format_size(size));
                    fs::remove_file(entry.path())?;
                }
                total_freed += size;
            }
        }
    }

    if !found_stale {
        println!("No stale archives found in deps/");
    }

    Ok(total_freed)
}
