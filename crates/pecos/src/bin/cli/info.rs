//! Implementation of the `info` command

#![allow(clippy::unnecessary_wraps)]

use pecos_build::Result;
use pecos_build::home::{get_cache_dir, get_deps_dir, get_llvm_dir, get_pecos_home};
use pecos_build::llvm::{find_llvm_14, get_llvm_version, get_repo_root_from_manifest};
use std::process::Command;

/// Run the info command
pub fn run() -> Result<()> {
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

    Ok(())
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
    let cuda_status = detect_cuda();
    println!("  CUDA:     {cuda_status}");

    // Python
    let python_status = detect_python();
    println!("  Python:   {python_status}");

    // uv
    let uv_status = detect_uv();
    println!("  uv:       {uv_status}");

    // Julia
    let julia_status = detect_julia();
    println!("  Julia:    {julia_status}");

    // Go
    let go_status = detect_go();
    println!("  Go:       {go_status}");
}

/// Detect CUDA availability
#[allow(clippy::collapsible_if)]
fn detect_cuda() -> String {
    // Check for nvcc
    if let Ok(output) = Command::new("nvcc").arg("--version").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Extract version from output like "Cuda compilation tools, release 12.0, V12.0.140"
            if let Some(line) = stdout.lines().find(|l| l.contains("release")) {
                if let Some(version) = line.split("release ").nth(1) {
                    if let Some(ver) = version.split(',').next() {
                        return format!("{ver} (nvcc found)");
                    }
                }
            }
            return "available (nvcc found)".to_string();
        }
    }

    // Check for CUDA_PATH environment variable
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        return format!("CUDA_PATH={cuda_path}");
    }

    "not detected".to_string()
}

/// Detect Python installation
#[allow(clippy::collapsible_if)]
fn detect_python() -> String {
    for cmd in ["python3", "python"] {
        if let Ok(output) = Command::new(cmd).arg("--version").output() {
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout);
                let version = version.trim();
                if version.is_empty() {
                    // Some systems output to stderr
                    let version = String::from_utf8_lossy(&output.stderr);
                    return version.trim().to_string();
                }
                return version.to_string();
            }
        }
    }
    "not found".to_string()
}

/// Detect uv installation
#[allow(clippy::collapsible_if)]
fn detect_uv() -> String {
    if let Ok(output) = Command::new("uv").arg("--version").output() {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            return version.trim().to_string();
        }
    }
    "not found".to_string()
}

/// Detect Julia installation
#[allow(clippy::collapsible_if)]
fn detect_julia() -> String {
    if let Ok(output) = Command::new("julia").arg("--version").output() {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            return version.trim().to_string();
        }
    }
    "not found".to_string()
}

/// Detect Go installation
#[allow(clippy::collapsible_if)]
fn detect_go() -> String {
    if let Ok(output) = Command::new("go").arg("version").output() {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            // Output is like "go version go1.21.0 linux/amd64"
            let version = version.trim();
            if let Some(ver) = version.strip_prefix("go version ") {
                return ver.to_string();
            }
            return version.to_string();
        }
    }
    "not found".to_string()
}
