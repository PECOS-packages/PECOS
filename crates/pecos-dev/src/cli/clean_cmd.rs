//! Implementation of the `clean` subcommand

use crate::Result;
use crate::errors::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Run the clean subcommand
pub fn run(command: super::CleanCommands) -> Result<()> {
    match command {
        super::CleanCommands::Build {
            dry_run,
            skip_cargo,
            verbose,
        } => run_build(dry_run, skip_cargo, verbose),
        super::CleanCommands::Deps { verbose } => run_deps(verbose),
        super::CleanCommands::Cache { verbose } => run_cache(verbose),
        super::CleanCommands::Llvm { verbose } => run_llvm(verbose),
        super::CleanCommands::Cuda { verbose } => run_cuda(verbose),
        super::CleanCommands::All {
            include_llvm,
            include_cuda,
            verbose,
        } => run_all(include_llvm, include_cuda, verbose),
    }
}

/// Get the repository root
fn get_repo_root() -> Result<PathBuf> {
    let mut current = std::env::current_dir()?;

    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            let content = fs::read_to_string(&cargo_toml)?;
            if content.contains("[workspace]") {
                return Ok(current);
            }
        }

        if !current.pop() {
            return Err(Error::Config(
                "Could not find PECOS repository root".to_string(),
            ));
        }
    }
}

/// Clean build artifacts cross-platform
#[allow(clippy::too_many_lines)]
fn run_build(dry_run: bool, skip_cargo: bool, verbose: u8) -> Result<()> {
    let repo_root = get_repo_root()?;

    if verbose >= 1 {
        println!("Cleaning build artifacts...");
    }

    let mut total_removed = 0;

    // 1. Remove top-level directories
    let top_level_dirs = ["dist", "site", ".ruff_cache"];
    for dir in top_level_dirs {
        total_removed += remove_path(&repo_root.join(dir), dry_run, verbose)?;
    }

    // 2. Remove *.egg-info directories at root
    total_removed += remove_glob(&repo_root, "*.egg-info", dry_run, verbose)?;

    // 3. Remove python/docs/_build
    total_removed += remove_path(&repo_root.join("python/docs/_build"), dry_run, verbose)?;

    // 4. Remove directories recursively by name
    let recursive_dirs = [
        (".", "build"),
        (".", ".pytest_cache"),
        (".", ".ipynb_checkpoints"),
        (".", ".hypothesis"),
        (".", "junit"),
        (".", "__pycache__"),
        ("crates", "target"),
        ("python", "target"),
    ];

    for (base, dir_name) in recursive_dirs {
        let base_path = repo_root.join(base);
        if base_path.exists() {
            total_removed += remove_dirs_recursive(&base_path, dir_name, dry_run, verbose)?;
        }
    }

    // 5. Remove compiled extensions in python/
    let python_dir = repo_root.join("python");
    if python_dir.exists() {
        total_removed += remove_files_recursive(&python_dir, &["*.so", "*.pyd"], dry_run, verbose)?;
    }

    // 6. Clean pecos_rslib from venv
    let venv_lib = repo_root.join(".venv/lib");
    if venv_lib.exists() {
        total_removed += remove_venv_package(&venv_lib, "pecos_rslib", dry_run, verbose)?;
    }

    // 7. Clean Julia artifacts
    let julia_manifests = [
        "julia/PECOS.jl/Manifest.toml",
        "julia/PECOS.jl/dev/PECOS_julia_jll/Manifest.toml",
        "julia/PECOS.jl/dev/PECOS_julia_jll/src/Manifest.toml",
    ];
    for manifest in julia_manifests {
        total_removed += remove_path(&repo_root.join(manifest), dry_run, verbose)?;
    }

    let julia_dir = repo_root.join("julia");
    if julia_dir.exists() {
        total_removed += remove_files_recursive(
            &julia_dir,
            &["*.jl.*.cov", "*.jl.cov", "*.jl.mem"],
            dry_run,
            verbose,
        )?;
    }

    // 8. Clean uv cache for pecos-rslib
    if dry_run && verbose >= 1 {
        println!("Would run: uv cache clean pecos-rslib");
    } else if !dry_run {
        let _ = Command::new("uv")
            .args(["cache", "clean", "pecos-rslib"])
            .output();
    }

    // 9. Run cargo clean (unless skipped)
    if !skip_cargo {
        if dry_run && verbose >= 1 {
            println!("Would run: cargo clean");
        } else if !dry_run {
            if verbose >= 1 {
                println!("Running cargo clean...");
            }
            let mut cmd = Command::new("cargo");
            cmd.arg("clean").current_dir(&repo_root);
            if verbose == 0 {
                cmd.arg("-q");
            }
            let status = cmd.status();
            if let Err(e) = status
                && verbose >= 1
            {
                eprintln!("Warning: cargo clean failed: {e}");
            }
        }
    }

    // Summary
    if verbose >= 1 {
        println!();
        if dry_run {
            println!("Dry run: {total_removed} items would be removed");
        } else {
            println!("Done. Removed {total_removed} items.");
        }
    }

    Ok(())
}

/// Clean ~/.pecos/deps/
fn run_deps(verbose: u8) -> Result<()> {
    let cleaned = clean_deps_internal(verbose)?;
    if !cleaned {
        if verbose >= 1 {
            println!("Nothing to clean (deps directory does not exist)");
        } else {
            println!("Nothing to clean");
        }
    }
    Ok(())
}

/// Internal helper that cleans deps and returns whether anything was cleaned
fn clean_deps_internal(verbose: u8) -> Result<bool> {
    use crate::home::get_deps_dir_path;

    let deps_dir = get_deps_dir_path()?;
    if deps_dir.exists() {
        if verbose >= 1 {
            println!("Removing: {}", deps_dir.display());
        }
        fs::remove_dir_all(&deps_dir)?;
        if verbose >= 1 {
            println!("Done.");
        } else {
            println!("Cleaned ~/.pecos/deps/");
        }
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Clean ~/.pecos/cache/ and tmp/
fn run_cache(verbose: u8) -> Result<()> {
    let cleaned = clean_cache_internal(verbose)?;
    if !cleaned {
        if verbose >= 1 {
            println!("Nothing to clean (cache/tmp directories do not exist)");
        } else {
            println!("Nothing to clean");
        }
    }
    Ok(())
}

/// Internal helper that cleans cache/tmp and returns whether anything was cleaned
fn clean_cache_internal(verbose: u8) -> Result<bool> {
    use crate::home::{get_cache_dir_path, get_tmp_dir_path};

    let cache_dir = get_cache_dir_path()?;
    let tmp_dir = get_tmp_dir_path()?;

    let mut cleaned = false;

    if cache_dir.exists() {
        if verbose >= 1 {
            println!("Removing: {}", cache_dir.display());
        }
        fs::remove_dir_all(&cache_dir)?;
        cleaned = true;
    }

    if tmp_dir.exists() {
        if verbose >= 1 {
            println!("Removing: {}", tmp_dir.display());
        }
        fs::remove_dir_all(&tmp_dir)?;
        cleaned = true;
    }

    if cleaned {
        if verbose >= 1 {
            println!("Done.");
        } else {
            println!("Cleaned ~/.pecos/cache/");
        }
    }
    Ok(cleaned)
}

/// Clean ~/.pecos/llvm/
fn run_llvm(verbose: u8) -> Result<()> {
    let cleaned = clean_llvm_internal(verbose)?;
    if !cleaned {
        if verbose >= 1 {
            println!("Nothing to clean (LLVM directory does not exist)");
        } else {
            println!("Nothing to clean");
        }
    }
    Ok(())
}

/// Internal helper that cleans LLVM and returns whether anything was cleaned
fn clean_llvm_internal(verbose: u8) -> Result<bool> {
    use crate::home::get_llvm_dir_path;

    let llvm_dir = get_llvm_dir_path()?;
    if llvm_dir.exists() {
        if verbose >= 1 {
            println!("Removing: {}", llvm_dir.display());
        }
        fs::remove_dir_all(&llvm_dir)?;
        if verbose >= 1 {
            println!("Done. Run 'pecos-dev llvm install' to reinstall LLVM.");
        } else {
            println!("Cleaned ~/.pecos/llvm/");
        }
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Clean ~/.pecos/cuda/
fn run_cuda(verbose: u8) -> Result<()> {
    let cleaned = clean_cuda_internal(verbose)?;
    if !cleaned {
        if verbose >= 1 {
            println!("Nothing to clean (CUDA directory does not exist)");
        } else {
            println!("Nothing to clean");
        }
    }
    Ok(())
}

/// Internal helper that cleans CUDA and returns whether anything was cleaned
fn clean_cuda_internal(verbose: u8) -> Result<bool> {
    use crate::cuda::get_pecos_cuda_dir;

    if let Some(cuda_dir) = get_pecos_cuda_dir()
        && cuda_dir.exists()
    {
        if verbose >= 1 {
            println!("Removing: {}", cuda_dir.display());
        }
        fs::remove_dir_all(&cuda_dir)?;
        if verbose >= 1 {
            println!("Done. Run 'pecos-dev cuda install' to reinstall CUDA.");
        } else {
            println!("Cleaned ~/.pecos/cuda/");
        }
        return Ok(true);
    }
    Ok(false)
}

/// Clean everything
fn run_all(include_llvm: bool, include_cuda: bool, verbose: u8) -> Result<()> {
    use crate::home::{get_cache_dir_path, get_deps_dir_path, get_llvm_dir_path, get_tmp_dir_path};

    let deps_dir = get_deps_dir_path()?;
    let cache_dir = get_cache_dir_path()?;
    let tmp_dir = get_tmp_dir_path()?;

    let mut cleaned = Vec::new();

    // Clean deps
    if deps_dir.exists() {
        if verbose >= 1 {
            println!("Removing: {}", deps_dir.display());
        }
        fs::remove_dir_all(&deps_dir)?;
        cleaned.push("deps");
    }

    // Clean cache
    if cache_dir.exists() {
        if verbose >= 1 {
            println!("Removing: {}", cache_dir.display());
        }
        fs::remove_dir_all(&cache_dir)?;
        cleaned.push("cache");
    }

    // Clean tmp
    if tmp_dir.exists() {
        if verbose >= 1 {
            println!("Removing: {}", tmp_dir.display());
        }
        fs::remove_dir_all(&tmp_dir)?;
        cleaned.push("tmp");
    }

    // Clean LLVM if requested
    if include_llvm {
        let llvm_dir = get_llvm_dir_path()?;
        if llvm_dir.exists() {
            if verbose >= 1 {
                println!("Removing: {}", llvm_dir.display());
            }
            fs::remove_dir_all(&llvm_dir)?;
            cleaned.push("llvm");
        }
    }

    // Clean CUDA if requested
    if include_cuda
        && let Some(cuda_dir) = crate::cuda::get_pecos_cuda_dir()
        && cuda_dir.exists()
    {
        if verbose >= 1 {
            println!("Removing: {}", cuda_dir.display());
        }
        fs::remove_dir_all(&cuda_dir)?;
        cleaned.push("cuda");
    }

    // Summary
    if cleaned.is_empty() {
        println!("Nothing to clean");
    } else if verbose >= 1 {
        println!("Done.");
    } else {
        println!("Cleaned ~/.pecos/{{{}}}", cleaned.join(","));
    }

    Ok(())
}

// Helper functions

/// Remove a path (file or directory)
fn remove_path(path: &Path, dry_run: bool, verbose: u8) -> Result<usize> {
    if !path.exists() {
        return Ok(0);
    }

    if dry_run {
        if verbose >= 1 {
            println!("Would remove: {}", path.display());
        }
    } else {
        if verbose >= 1 {
            println!("Removing: {}", path.display());
        }
        if path.is_dir() {
            fs::remove_dir_all(path)?;
        } else {
            fs::remove_file(path)?;
        }
    }
    Ok(1)
}

/// Remove directories matching a glob pattern at a path
fn remove_glob(base: &Path, pattern: &str, dry_run: bool, verbose: u8) -> Result<usize> {
    let mut count = 0;

    if let Ok(entries) = fs::read_dir(base) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if glob_match(pattern, &name) && entry.path().is_dir() {
                count += remove_path(&entry.path(), dry_run, verbose)?;
            }
        }
    }

    Ok(count)
}

/// Simple glob matching (only supports * wildcard)
fn glob_match(pattern: &str, name: &str) -> bool {
    if let Some(suffix) = pattern.strip_prefix('*') {
        name.ends_with(suffix)
    } else if let Some(prefix) = pattern.strip_suffix('*') {
        name.starts_with(prefix)
    } else {
        pattern == name
    }
}

/// Remove directories with a specific name recursively
fn remove_dirs_recursive(base: &Path, dir_name: &str, dry_run: bool, verbose: u8) -> Result<usize> {
    let mut count = 0;
    let mut dirs_to_remove = Vec::new();

    collect_dirs_by_name(base, dir_name, &mut dirs_to_remove);

    for dir in dirs_to_remove {
        count += remove_path(&dir, dry_run, verbose)?;
    }

    Ok(count)
}

/// Collect directories with a specific name
fn collect_dirs_by_name(base: &Path, dir_name: &str, result: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(base) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name == dir_name {
                    result.push(path);
                } else if name != ".git" && name != "node_modules" {
                    collect_dirs_by_name(&path, dir_name, result);
                }
            }
        }
    }
}

/// Remove files matching patterns recursively
fn remove_files_recursive(
    base: &Path,
    patterns: &[&str],
    dry_run: bool,
    verbose: u8,
) -> Result<usize> {
    let mut count = 0;
    let mut files_to_remove = Vec::new();

    collect_files_by_pattern(base, patterns, &mut files_to_remove);

    for file in files_to_remove {
        count += remove_path(&file, dry_run, verbose)?;
    }

    Ok(count)
}

/// Collect files matching patterns
fn collect_files_by_pattern(base: &Path, patterns: &[&str], result: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(base) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let name = entry.file_name().to_string_lossy().to_string();
                for pattern in patterns {
                    if glob_match(pattern, &name) {
                        result.push(path.clone());
                        break;
                    }
                }
            } else if path.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name != ".git" && name != "node_modules" && name != "target" {
                    collect_files_by_pattern(&path, patterns, result);
                }
            }
        }
    }
}

/// Remove a package from venv
fn remove_venv_package(
    venv_lib: &Path,
    package: &str,
    dry_run: bool,
    verbose: u8,
) -> Result<usize> {
    let mut count = 0;

    if let Ok(entries) = fs::read_dir(venv_lib) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("python") {
                    let site_packages = path.join("site-packages");
                    if site_packages.exists() {
                        let pkg_dir = site_packages.join(package);
                        count += remove_path(&pkg_dir, dry_run, verbose)?;

                        if let Ok(sp_entries) = fs::read_dir(&site_packages) {
                            for sp_entry in sp_entries.flatten() {
                                let sp_name = sp_entry.file_name().to_string_lossy().to_string();
                                if sp_name.starts_with(package)
                                    && sp_name.contains(".dist-info")
                                    && sp_entry.path().is_dir()
                                {
                                    count += remove_path(&sp_entry.path(), dry_run, verbose)?;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(count)
}
