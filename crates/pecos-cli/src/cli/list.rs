//! Implementation of the `list` command

use pecos_build::cuda::find_cuda;
use pecos_build::cuquantum::find_cuquantum;
use pecos_build::deps::list_dependencies;
use pecos_build::home::{get_cache_dir, get_deps_dir};
use pecos_build::llvm::{find_llvm, get_llvm_version, get_repo_root_from_manifest};
use std::fs;
use std::path::Path;

/// Run the list command
pub fn run(verbose: bool) {
    println!("PECOS Dependencies");
    println!("==================");
    println!();

    // LLVM status
    let repo_root = get_repo_root_from_manifest();
    if let Some(llvm_path) = find_llvm(repo_root) {
        let version = get_llvm_version(&llvm_path)
            .map(|v| format!(" ({v})"))
            .unwrap_or_default();
        println!("LLVM 21.1:     {}{version}", llvm_path.display());
    } else if pecos_build::llvm::installer::managed_install_unavailable_reason().is_some() {
        println!("LLVM 21.1:     not found (configure shared LLVM 21 manually)");
    } else {
        println!("LLVM 21.1:     not found (install with: pecos install llvm)");
    }

    // CUDA status
    if let Some(cuda_path) = find_cuda() {
        println!("CUDA:        {}", cuda_path.display());
    } else {
        println!("CUDA:        not found");
    }

    // cuQuantum status
    if let Some(cuq_path) = find_cuquantum() {
        println!("cuQuantum:   {}", cuq_path.display());
    } else {
        println!("cuQuantum:   not found");
    }
    println!();

    if verbose {
        // Extracted deps with sizes
        println!("Installed (~/.pecos/deps/):");
        if let Ok(deps_dir) = get_deps_dir() {
            if deps_dir.exists() {
                let mut entries: Vec<_> = fs::read_dir(&deps_dir)
                    .ok()
                    .into_iter()
                    .flatten()
                    .flatten()
                    .filter(|e| e.path().is_dir())
                    .collect();
                entries.sort_by_key(std::fs::DirEntry::file_name);

                if entries.is_empty() {
                    println!("  (none)");
                } else {
                    for entry in &entries {
                        let size = dir_size_display(&entry.path());
                        println!("  {:30} {size}", entry.file_name().to_string_lossy());
                    }
                }
            } else {
                println!("  (not created yet)");
            }
        }
        println!();

        // Cached downloads with sizes
        println!("Cached downloads (~/.pecos/cache/):");
        if let Ok(cache_dir) = get_cache_dir() {
            if cache_dir.exists() {
                let mut entries: Vec<_> = fs::read_dir(&cache_dir)
                    .ok()
                    .into_iter()
                    .flatten()
                    .flatten()
                    .filter(|e| e.path().is_file())
                    .collect();
                entries.sort_by_key(std::fs::DirEntry::file_name);

                if entries.is_empty() {
                    println!("  (none)");
                } else {
                    for entry in &entries {
                        let size = entry
                            .metadata()
                            .map_or_else(|_| "?".to_string(), |m| format_bytes(m.len()));
                        println!("  {:50} {size}", entry.file_name().to_string_lossy());
                    }
                }
            } else {
                println!("  (not created yet)");
            }
        }
    } else {
        // Non-verbose: just list available deps from manifest
        println!("Available Dependencies:");
        for dep in list_dependencies() {
            println!("  {}: {} - {}", dep.name, dep.version, dep.description);
        }
    }
}

fn dir_size_display(path: &Path) -> String {
    dir_size(path).map_or_else(|| "?".to_string(), format_bytes)
}

#[allow(clippy::cast_precision_loss)] // byte count as f64 for display
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.0} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn dir_size(path: &Path) -> Option<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(path).ok()? {
        let entry = entry.ok()?;
        let meta = entry.metadata().ok()?;
        if meta.is_dir() {
            total += dir_size(&entry.path())?;
        } else {
            total += meta.len();
        }
    }
    Some(total)
}
