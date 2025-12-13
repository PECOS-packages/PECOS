//! Implementation of the `info` command

#![allow(clippy::unnecessary_wraps)]

use crate::Result;
use crate::home::{get_cache_dir, get_deps_dir, get_llvm_dir, get_pecos_home};

/// Run the info command
pub fn run() -> Result<()> {
    println!("PECOS Dependency Management");
    println!("============================");
    println!();

    // Show PECOS home directory
    match get_pecos_home() {
        Ok(home) => {
            println!("PECOS Home: {}", home.display());
            if home.exists() {
                println!("  Status: exists");
            } else {
                println!("  Status: not created yet");
            }
        }
        Err(e) => {
            println!("PECOS Home: <error: {e}>");
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
