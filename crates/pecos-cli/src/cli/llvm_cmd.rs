//! Implementation of LLVM subcommands

use super::LlvmCommands;
use pecos_build::Result;
use pecos_build::llvm::config::{auto_configure_llvm, validate_llvm_config, write_cargo_config};
use pecos_build::llvm::{
    LLVM_SYS_PREFIX_ENV, REQUIRED_VERSION, find_cargo_project_root,
    find_configured_or_detected_llvm, find_llvm, find_tool, get_llvm_shared_libraries,
    get_llvm_shared_mode, get_llvm_version, get_pecos_command, get_repo_root_from_manifest,
    is_valid_llvm,
};
use std::path::Path;

/// Run an LLVM subcommand
pub fn run(command: LlvmCommands) -> Result<()> {
    match command {
        LlvmCommands::Check { quiet } => {
            run_check(quiet);
            Ok(())
        }
        LlvmCommands::Ensure {
            managed,
            no_configure,
        } => run_ensure(managed, no_configure),
        LlvmCommands::Configure { path } => run_configure(path),
        LlvmCommands::Find { export } => {
            run_find(export);
            Ok(())
        }
        LlvmCommands::Version => run_version(),
        LlvmCommands::Validate { path } => run_validate(path),
        LlvmCommands::Tool { name } => {
            run_tool(&name);
            Ok(())
        }
    }
}

fn run_check(quiet: bool) {
    let repo_root = get_repo_root_from_manifest();
    if let Some(llvm_path) = find_configured_or_detected_llvm(repo_root) {
        if !quiet {
            println!("LLVM 21.1 found at: {}", llvm_path.display());
            if let Ok(version) = get_llvm_version(&llvm_path) {
                println!("Version: {version}");
            }
            print_link_info(&llvm_path);

            // Validate configuration
            let validation = validate_llvm_config();
            validation.print_warnings();

            // Exit with error if config is unhealthy (would cause build failures)
            if !validation.is_healthy() && validation.configured_path.is_some() {
                std::process::exit(1);
            }
        }
    } else {
        if !quiet {
            let cmd = get_pecos_command();
            eprintln!("LLVM 21.1 not found");
            eprintln!();
            if let Some(reason) = pecos_build::llvm::installer::managed_install_unavailable_reason()
            {
                eprintln!("{reason}");
                eprintln!();
                eprintln!("After installing LLVM 21, configure it with:");
                eprintln!("  {cmd} llvm configure /path/to/llvm");
            } else {
                eprintln!("Install with: `{cmd} install llvm`");
            }
        }
        std::process::exit(1);
    }
}

fn run_ensure(managed: bool, no_configure: bool) -> Result<()> {
    let llvm_path = if managed {
        ensure_managed_llvm(no_configure)?
    } else if let Some(path) = find_llvm(get_repo_root_from_manifest()) {
        if !no_configure {
            auto_configure_llvm(None)?;
        }
        path
    } else {
        pecos_build::llvm::installer::install_llvm(false, no_configure)?
    };

    let version = get_llvm_version(&llvm_path)?;
    println!("LLVM {version} ready at {}", llvm_path.display());
    Ok(())
}

fn ensure_managed_llvm(no_configure: bool) -> Result<std::path::PathBuf> {
    let llvm_path =
        pecos_build::home::get_versioned_dep_path("llvm", pecos_build::home::LLVM_VERSION)?;

    if !pecos_build::llvm::installer::is_valid_installation(&llvm_path) {
        if llvm_path.exists() {
            println!(
                "Existing LLVM at {} failed runtime validation; reinstalling.",
                llvm_path.display()
            );
            pecos_build::llvm::installer::install_llvm(true, no_configure)?;
        } else {
            pecos_build::llvm::installer::install_llvm(false, no_configure)?;
        }
    } else if !no_configure {
        auto_configure_llvm(None)?;
    }

    Ok(llvm_path)
}

fn run_configure(path: Option<String>) -> Result<()> {
    let llvm_path = if let Some(path) = path {
        let input_path = std::path::PathBuf::from(&path);
        let llvm_path = input_path.canonicalize().map_err(|e| {
            pecos_build::errors::Error::Llvm(format!(
                "Could not resolve LLVM path {}: {e}",
                input_path.display()
            ))
        })?;
        if !is_valid_llvm(&llvm_path) {
            return Err(pecos_build::errors::Error::Llvm(format!(
                "{} is not a valid LLVM {REQUIRED_VERSION} installation",
                llvm_path.display()
            )));
        }

        let project_root = get_repo_root_from_manifest()
            .or_else(find_cargo_project_root)
            .ok_or_else(|| {
                pecos_build::errors::Error::Config("Could not find Cargo project root".into())
            })?;
        write_cargo_config(&project_root, &llvm_path, true)?;
        llvm_path
    } else {
        auto_configure_llvm(None)?
    };

    println!("Configured LLVM path: {}", llvm_path.display());
    println!("Updated .cargo/config.toml");
    Ok(())
}

fn run_find(export: bool) {
    let repo_root = get_repo_root_from_manifest();
    if let Some(llvm_path) = find_configured_or_detected_llvm(repo_root) {
        if export {
            println!("export {LLVM_SYS_PREFIX_ENV}=\"{}\"", llvm_path.display());
        } else {
            println!("{}", llvm_path.display());
        }
    } else {
        eprintln!("LLVM 21.1 not found");
        std::process::exit(1);
    }
}

fn run_version() -> Result<()> {
    let repo_root = get_repo_root_from_manifest();
    if let Some(llvm_path) = find_configured_or_detected_llvm(repo_root) {
        let version = get_llvm_version(&llvm_path)?;
        println!("LLVM version: {version}");
        println!("Location: {}", llvm_path.display());
        Ok(())
    } else {
        eprintln!("LLVM 21.1 not found");
        std::process::exit(1);
    }
}

fn run_validate(path: Option<String>) -> Result<()> {
    let llvm_path = if let Some(p) = path {
        std::path::PathBuf::from(p)
    } else {
        let repo_root = get_repo_root_from_manifest();
        find_configured_or_detected_llvm(repo_root).ok_or_else(|| {
            pecos_build::errors::Error::Llvm(
                "LLVM 21.1 not found. Specify a path or install first.".into(),
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
        if pecos_build::llvm::is_required_llvm_version(&version) {
            println!("Version: {version} [OK]");
        } else {
            println!("Version: {version} [WARNING: expected {REQUIRED_VERSION}]");
            all_present = false;
        }
    } else {
        println!("Version: could not determine [ERROR]");
        all_present = false;
    }

    print_link_info(&llvm_path);
    println!();

    println!();
    if all_present {
        println!("Validation: PASSED");
    } else {
        println!("Validation: FAILED");
        std::process::exit(1);
    }

    Ok(())
}

fn run_tool(name: &str) {
    if let Some(tool_path) = find_tool(name) {
        println!("{}", tool_path.display());
    } else {
        eprintln!("Tool '{name}' not found");
        std::process::exit(1);
    }
}

fn print_link_info(llvm_path: &Path) {
    if let Ok(mode) = get_llvm_shared_mode(llvm_path) {
        println!("Link mode: {mode}");
    }

    if let Some(libraries) = get_llvm_shared_libraries(llvm_path) {
        println!("Shared library: {libraries}");
    } else {
        println!("Shared library: unavailable");
    }
}
