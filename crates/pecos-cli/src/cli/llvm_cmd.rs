//! Implementation of LLVM subcommands

#![allow(clippy::unnecessary_wraps)]

use super::LlvmCommands;
use pecos_build::Result;
use pecos_build::llvm::config::{auto_configure_llvm, validate_llvm_config};
use pecos_build::llvm::{
    find_llvm_14, find_tool, get_llvm_version, get_pecos_command, get_repo_root_from_manifest,
};

/// Run an LLVM subcommand
pub fn run(command: LlvmCommands) -> Result<()> {
    match command {
        LlvmCommands::Check { quiet } => run_check(quiet),
        LlvmCommands::Configure => run_configure(),
        LlvmCommands::Find { export } => run_find(export),
        LlvmCommands::Version => run_version(),
        LlvmCommands::Validate { path } => run_validate(path),
        LlvmCommands::Tool { name } => run_tool(&name),
    }
}

fn run_check(quiet: bool) -> Result<()> {
    let repo_root = get_repo_root_from_manifest();
    if let Some(llvm_path) = find_llvm_14(repo_root) {
        if !quiet {
            println!("LLVM 14 found at: {}", llvm_path.display());
            if let Ok(version) = get_llvm_version(&llvm_path) {
                println!("Version: {version}");
            }

            // Validate configuration
            let validation = validate_llvm_config();
            validation.print_warnings();

            // Exit with error if config is unhealthy (would cause build failures)
            if !validation.is_healthy() && validation.configured_path.is_some() {
                std::process::exit(1);
            }
        }
        Ok(())
    } else {
        if !quiet {
            let cmd = get_pecos_command();
            eprintln!("LLVM 14 not found");
            eprintln!();
            eprintln!("Install with: `{cmd} install llvm`");
        }
        std::process::exit(1);
    }
}

fn run_configure() -> Result<()> {
    let llvm_path = auto_configure_llvm(None)?;
    println!("Configured LLVM path: {}", llvm_path.display());
    println!("Updated .cargo/config.toml");
    Ok(())
}

fn run_find(export: bool) -> Result<()> {
    let repo_root = get_repo_root_from_manifest();
    if let Some(llvm_path) = find_llvm_14(repo_root) {
        if export {
            println!("export LLVM_SYS_140_PREFIX=\"{}\"", llvm_path.display());
        } else {
            println!("{}", llvm_path.display());
        }
        Ok(())
    } else {
        eprintln!("LLVM 14 not found");
        std::process::exit(1);
    }
}

fn run_version() -> Result<()> {
    let repo_root = get_repo_root_from_manifest();
    if let Some(llvm_path) = find_llvm_14(repo_root) {
        let version = get_llvm_version(&llvm_path)?;
        println!("LLVM version: {version}");
        println!("Location: {}", llvm_path.display());
        Ok(())
    } else {
        eprintln!("LLVM 14 not found");
        std::process::exit(1);
    }
}

fn run_validate(path: Option<String>) -> Result<()> {
    let llvm_path = if let Some(p) = path {
        std::path::PathBuf::from(p)
    } else {
        let repo_root = get_repo_root_from_manifest();
        find_llvm_14(repo_root).ok_or_else(|| {
            pecos_build::errors::Error::Llvm(
                "LLVM 14 not found. Specify a path or install first.".into(),
            )
        })?
    };

    println!("Validating LLVM installation at: {}", llvm_path.display());
    println!();

    // Check basic structure
    let exe_ext = if cfg!(windows) { ".exe" } else { "" };
    let required_tools = [
        format!("bin/llvm-config{exe_ext}"),
        format!("bin/clang{exe_ext}"),
        format!("bin/llvm-as{exe_ext}"),
        format!("bin/llvm-dis{exe_ext}"),
        format!("bin/opt{exe_ext}"),
    ];

    let mut all_present = true;
    for tool in &required_tools {
        let tool_path = llvm_path.join(tool);
        if tool_path.exists() {
            println!("  [OK] {tool}");
        } else {
            println!("  [MISSING] {tool}");
            all_present = false;
        }
    }

    // Check version
    println!();
    if let Ok(version) = get_llvm_version(&llvm_path) {
        if version.starts_with("14.") {
            println!("Version: {version} [OK]");
        } else {
            println!("Version: {version} [WARNING: expected 14.x]");
            all_present = false;
        }
    } else {
        println!("Version: could not determine [ERROR]");
        all_present = false;
    }

    println!();
    if all_present {
        println!("Validation: PASSED");
    } else {
        println!("Validation: FAILED");
        std::process::exit(1);
    }

    Ok(())
}

fn run_tool(name: &str) -> Result<()> {
    if let Some(tool_path) = find_tool(name) {
        println!("{}", tool_path.display());
        Ok(())
    } else {
        eprintln!("Tool '{name}' not found");
        std::process::exit(1);
    }
}
