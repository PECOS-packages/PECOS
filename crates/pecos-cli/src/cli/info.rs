//! Implementation of the `info` command

use pecos_build::cuda::{find_cuda, get_cuda_version};
use pecos_build::cuquantum::{find_cuquantum, get_cuquantum_version};
use pecos_build::home::{get_cache_dir, get_deps_dir, get_llvm_dir, get_pecos_home};
use pecos_build::llvm::{find_llvm_14, get_llvm_version, get_repo_root_from_manifest};
use std::process::Command;

/// Run the info command
pub fn run() {
    println!("PECOS Development Environment");
    println!("==============================");
    println!();

    // Show system toolchain status
    print_toolchain_status();
    println!();

    // Show PECOS home directory
    println!("PECOS Home:");
    match get_pecos_home() {
        Ok(home) => {
            println!("  Path: {}", home.display());
            if home.exists() {
                println!("  Status: exists");
            } else {
                println!("  Status: not created yet");
            }
        }
        Err(e) => {
            println!("  Path: <error: {e}>");
        }
    }
    println!();

    // Show subdirectories
    println!("Subdirectories:");

    if let Ok(llvm_dir) = get_llvm_dir() {
        print!("  LLVM:  {}", llvm_dir.display());
        if llvm_dir.exists() {
            println!(" (exists)");
        } else {
            println!(" (not installed)");
        }
    }

    if let Ok(cuda_dir) = pecos_build::home::get_cuda_dir() {
        print!("  CUDA:  {}", cuda_dir.display());
        if cuda_dir.exists() {
            println!(" (exists)");
        } else {
            println!(" (not installed)");
        }
    }

    if let Ok(cuq_dir) = pecos_build::home::get_cuquantum_dir() {
        print!("  cuQ:   {}", cuq_dir.display());
        if cuq_dir.exists() {
            println!(" (exists)");
        } else {
            println!(" (not installed)");
        }
    }

    if let Ok(deps_dir) = get_deps_dir() {
        print!("  Deps:  {}", deps_dir.display());
        if deps_dir.exists() {
            println!(" (exists)");
        } else {
            println!(" (empty)");
        }
    }

    if let Ok(cache_dir) = get_cache_dir() {
        print!("  Cache: {}", cache_dir.display());
        if cache_dir.exists() {
            println!(" (exists)");
        } else {
            println!(" (empty)");
        }
    }

    println!();

    // Show environment overrides if set
    println!("Environment Overrides:");
    let mut has_overrides = false;

    if let Ok(val) = std::env::var("PECOS_HOME") {
        println!("  PECOS_HOME = {val}");
        has_overrides = true;
    }
    if let Ok(val) = std::env::var("PECOS_DEPS_DIR") {
        println!("  PECOS_DEPS_DIR = {val}");
        has_overrides = true;
    }
    if let Ok(val) = std::env::var("PECOS_CACHE_DIR") {
        println!("  PECOS_CACHE_DIR = {val}");
        has_overrides = true;
    }
    if let Ok(val) = std::env::var("LLVM_SYS_140_PREFIX") {
        println!("  LLVM_SYS_140_PREFIX = {val}");
        has_overrides = true;
    }

    if !has_overrides {
        println!("  (none)");
    }
}

/// Print toolchain and dependency status
fn print_toolchain_status() {
    println!("Toolchain Status:");

    // LLVM
    let repo_root = get_repo_root_from_manifest();
    if let Some(llvm_path) = find_llvm_14(repo_root) {
        let version = get_llvm_version(&llvm_path).unwrap_or_else(|_| "unknown".to_string());
        println!("  LLVM 14:  {} ({})", version, llvm_path.display());
    } else {
        println!("  LLVM 14:  not found");
    }

    // CUDA
    if let Some(cuda_path) = find_cuda() {
        let version = get_cuda_version(&cuda_path).unwrap_or_else(|_| "unknown".to_string());
        println!("  CUDA:     {} ({})", version, cuda_path.display());
    } else {
        println!("  CUDA:     not found");
    }

    // cuQuantum
    if let Some(cuq_path) = find_cuquantum() {
        let version = get_cuquantum_version(&cuq_path).unwrap_or_else(|_| "unknown".to_string());
        println!("  cuQuantum: {} ({})", version, cuq_path.display());
    } else {
        println!("  cuQuantum: not found");
    }

    println!("  Python:   {}", detect_tool("python3", "--version"));
    println!("  uv:       {}", detect_tool("uv", "--version"));
    println!("  Julia:    {}", detect_tool("julia", "--version"));
    println!("  Go:       {}", detect_tool("go", "version"));
}

fn detect_tool(cmd: &str, arg: &str) -> String {
    let output = Command::new(cmd).arg(arg).output().ok();
    match output {
        Some(o) if o.status.success() => {
            let out = String::from_utf8_lossy(&o.stdout);
            let out = out.trim();
            if out.is_empty() {
                // Some tools (python) output to stderr
                String::from_utf8_lossy(&o.stderr).trim().to_string()
            } else {
                out.strip_prefix("go version ").unwrap_or(out).to_string()
            }
        }
        _ => "not found".to_string(),
    }
}
