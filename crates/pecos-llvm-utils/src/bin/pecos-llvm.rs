#!/usr/bin/env rust
//! PECOS LLVM management tool
//!
//! Handles LLVM 14 detection, installation, and configuration for PECOS.

use clap::{Parser, Subcommand};
use pecos_llvm_utils::{
    find_llvm_14, find_tool, get_repo_root_from_manifest, print_llvm_not_found_error,
};
use std::process;

#[derive(Parser)]
#[command(name = "pecos-llvm")]
#[command(about = "PECOS LLVM management tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Find LLVM 14 installation and print its path
    Find {
        /// Print export command for shell evaluation
        #[arg(long)]
        export: bool,
    },
    /// Check if LLVM 14 is available (exit code 0 if found, 1 if not)
    Check {
        /// Suppress output messages
        #[arg(short, long)]
        quiet: bool,
    },
    /// Install LLVM 14.0.6 to ~/.pecos/llvm/
    Install {
        /// Force reinstall even if already present
        #[arg(short, long)]
        force: bool,
        /// Skip automatic configuration after installation
        #[arg(long)]
        no_configure: bool,
    },
    /// Auto-configure LLVM for PECOS (updates .cargo/config.toml)
    Configure,
    /// Show LLVM version information
    Version,
    /// Validate an LLVM installation at a specific path
    Validate {
        /// Path to the LLVM installation to validate
        path: std::path::PathBuf,
    },
    /// Find a specific LLVM tool (e.g., llvm-as, clang)
    Tool {
        /// Name of the tool to find (e.g., "llvm-as", "clang", "llvm-link")
        name: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Find { export } => cmd_find(export),
        Commands::Check { quiet } => cmd_check(quiet),
        Commands::Install {
            force,
            no_configure,
        } => {
            cmd_install(force, no_configure);
        }
        Commands::Configure => cmd_configure(),
        Commands::Version => cmd_version(),
        Commands::Validate { path } => cmd_validate(&path),
        Commands::Tool { name } => cmd_tool(&name),
    }
}

fn cmd_find(export: bool) {
    let repo_root = get_repo_root_from_manifest();
    let llvm_path = find_llvm_14(repo_root);

    if let Some(path) = llvm_path {
        let path_str = path.to_string_lossy();
        if export {
            println!("export LLVM_SYS_140_PREFIX=\"{path_str}\"");
        } else {
            println!("{path_str}");
        }
        process::exit(0);
    } else {
        print_llvm_not_found_error();
        process::exit(1);
    }
}

fn cmd_check(quiet: bool) {
    let repo_root = get_repo_root_from_manifest();
    let llvm_path = find_llvm_14(repo_root);

    if let Some(path) = llvm_path {
        if !quiet {
            eprintln!("LLVM 14 found at: {}", path.display());
        }
        process::exit(0);
    } else {
        if !quiet {
            eprintln!("LLVM 14 not found");
        }
        process::exit(1);
    }
}

fn cmd_install(force: bool, no_configure: bool) {
    use pecos_llvm_utils::installer::install_llvm;

    match install_llvm(force, no_configure) {
        Ok(_install_path) => {
            // Success message is printed by install_llvm
            process::exit(0);
        }
        Err(e) => {
            eprintln!("Failed to install LLVM: {e}");
            process::exit(1);
        }
    }
}

fn cmd_configure() {
    use pecos_llvm_utils::auto_configure_llvm;

    println!("Auto-configuring LLVM for PECOS...");
    println!();

    match auto_configure_llvm(None) {
        Ok(configured_path) => {
            println!("Success! LLVM configured at: {}", configured_path.display());
            println!();
            println!("Updated .cargo/config.toml with LLVM configuration.");
            println!();
            println!("You can now build PECOS:");
            println!("  cargo build");
            process::exit(0);
        }
        Err(e) => {
            eprintln!("Failed to configure LLVM: {e}");
            eprintln!();
            eprintln!("To install LLVM, run:");
            eprintln!("  pecos-llvm install");
            process::exit(1);
        }
    }
}

fn cmd_version() {
    let repo_root = get_repo_root_from_manifest();
    let llvm_path = find_llvm_14(repo_root);

    if let Some(path) = llvm_path {
        println!("LLVM 14 found at: {}", path.display());

        // Try to get version from llvm-config
        let llvm_config = if cfg!(windows) {
            path.join("bin").join("llvm-config.exe")
        } else {
            path.join("bin").join("llvm-config")
        };

        if let Ok(output) = std::process::Command::new(&llvm_config)
            .arg("--version")
            .output()
            && output.status.success()
        {
            let version = String::from_utf8_lossy(&output.stdout);
            println!("Version: {}", version.trim());
        }
    } else {
        println!("LLVM 14 not found");
        process::exit(1);
    }
}

fn cmd_validate(path: &std::path::Path) {
    use pecos_llvm_utils::installer::{is_valid_installation, verify_llvm_runtime};

    println!("Validating LLVM installation at: {}", path.display());
    println!();

    // Check if path exists
    if !path.exists() {
        eprintln!("ERROR: Path does not exist");
        process::exit(1);
    }

    // Validate file structure
    println!("Checking file structure...");
    let files_valid = is_valid_installation(path);

    if !files_valid {
        eprintln!();
        eprintln!("ERROR: Validation FAILED: Missing critical files");
        eprintln!();
        eprintln!("This LLVM installation is incomplete or corrupted.");
        eprintln!("Consider reinstalling LLVM:");
        eprintln!("  cargo run -p pecos-llvm-utils --bin pecos-llvm -- install --force");
        process::exit(1);
    }

    println!("File structure OK");
    println!();

    // Validate runtime
    match verify_llvm_runtime(path) {
        Ok(()) => {
            println!();
            println!("All checks passed!");
            println!("This LLVM installation appears to be valid and functional.");
            process::exit(0);
        }
        Err(e) => {
            eprintln!();
            eprintln!("ERROR: Runtime validation FAILED: {e}");
            eprintln!();
            eprintln!("The LLVM binaries may be corrupted or have missing dependencies.");
            eprintln!("Consider reinstalling LLVM:");
            eprintln!("  cargo run -p pecos-llvm-utils --bin pecos-llvm -- install --force");
            process::exit(1);
        }
    }
}

fn cmd_tool(tool_name: &str) {
    if let Some(tool_path) = find_tool(tool_name) {
        println!("{}", tool_path.display());
        process::exit(0);
    } else {
        eprintln!("ERROR: Tool '{tool_name}' not found");
        eprintln!();
        eprintln!("Make sure LLVM 14 is installed:");
        eprintln!("  cargo run -p pecos-llvm-utils --bin pecos-llvm -- check");
        process::exit(1);
    }
}
