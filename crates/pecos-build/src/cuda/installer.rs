//! CUDA Toolkit installation functionality
//!
//! Downloads and installs CUDA Toolkit to `~/.pecos/deps/cuda/`

#![allow(clippy::case_sensitive_file_extension_comparisons)]

use crate::errors::{Error, Result};
use sevenz_rust::{Password, SevenZReader};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use super::{CUDA_VERSION, get_pecos_cuda_dir, is_valid_cuda_installation};

/// CUDA Toolkit download information
struct CudaDownload {
    url: String,
    filename: String,
    sha256: Option<&'static str>,
}

/// Get download URL for the current platform
fn get_download_info() -> Result<CudaDownload> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    match (os, arch) {
        ("linux", "x86_64") => Ok(CudaDownload {
            url: format!(
                "https://developer.download.nvidia.com/compute/cuda/{CUDA_VERSION}/local_installers/cuda_{CUDA_VERSION}_560.35.05_linux.run"
            ),
            filename: format!("cuda_{CUDA_VERSION}_560.35.05_linux.run"),
            // SHA256 can be added once verified
            sha256: None,
        }),
        ("linux", "aarch64") => Ok(CudaDownload {
            url: format!(
                "https://developer.download.nvidia.com/compute/cuda/{CUDA_VERSION}/local_installers/cuda_{CUDA_VERSION}_560.35.05_linux_sbsa.run"
            ),
            filename: format!("cuda_{CUDA_VERSION}_560.35.05_linux_sbsa.run"),
            sha256: None,
        }),
        ("windows", "x86_64") => Ok(CudaDownload {
            url: format!(
                "https://developer.download.nvidia.com/compute/cuda/{CUDA_VERSION}/local_installers/cuda_{CUDA_VERSION}_561.17_windows.exe"
            ),
            filename: format!("cuda_{CUDA_VERSION}_561.17_windows.exe"),
            sha256: None,
        }),
        ("macos", _) => Err(Error::Cuda(
            "CUDA is not supported on macOS (deprecated by NVIDIA since macOS 10.14)".into(),
        )),
        _ => Err(Error::Cuda(format!("Unsupported platform: {os}/{arch}"))),
    }
}

/// Install CUDA Toolkit to `~/.pecos/deps/cuda/`
///
/// # Arguments
/// * `force` - Force reinstall even if already present
///
/// # Errors
/// Returns an error if:
/// - Home directory cannot be determined
/// - CUDA is already installed (unless `force` is true)
/// - Platform is unsupported
/// - Download or extraction fails
/// - Installation verification fails
pub fn install_cuda(force: bool) -> Result<PathBuf> {
    let cuda_dir = get_pecos_cuda_dir()?;

    // Check if already installed
    if !force && cuda_dir.exists() && is_valid_cuda_installation(&cuda_dir) {
        return Err(Error::Cuda(
            "CUDA is already installed. Use --force to reinstall.".into(),
        ));
    }

    // Remove existing if force
    if force && cuda_dir.exists() {
        println!("Removing existing CUDA installation...");
        fs::remove_dir_all(&cuda_dir)?;
    }

    let download_info = get_download_info()?;

    println!("Installing CUDA Toolkit {CUDA_VERSION}...");
    println!("This will download ~4GB and may take 10-30 minutes depending on your connection.");
    println!();

    // Create cache directory
    let cache_dir = cuda_dir
        .parent()
        .ok_or_else(|| Error::Cuda("Invalid CUDA directory".into()))?
        .join("cache");
    fs::create_dir_all(&cache_dir)?;

    let archive_path = cache_dir.join(&download_info.filename);

    // Download if not already cached
    if archive_path.exists() {
        println!("Using cached download: {}", archive_path.display());
    } else {
        download_cuda(&download_info.url, &archive_path)?;

        // Verify checksum if available
        if let Some(expected_sha256) = download_info.sha256 {
            verify_checksum(&archive_path, expected_sha256)?;
        }
    }

    // Extract CUDA
    extract_cuda(&archive_path, &cuda_dir)?;

    // Verify installation
    if !is_valid_cuda_installation(&cuda_dir) {
        return Err(Error::Cuda(
            "Installation completed but verification failed".into(),
        ));
    }

    // Write version marker
    let version_file = cuda_dir.join("version.txt");
    fs::write(
        &version_file,
        format!("CUDA {CUDA_VERSION}\nInstalled by pecos\n"),
    )?;

    println!();
    println!("Installation complete!");
    println!(
        "CUDA Toolkit {} installed to: {}",
        CUDA_VERSION,
        cuda_dir.display()
    );
    println!();
    println!("To use this installation, you can either:");
    println!("  1. Build with pecos (automatically detected)");
    println!("  2. Set environment variables:");
    println!("     export CUDA_PATH=\"{}\"", cuda_dir.display());
    println!("     export PATH=\"{}/bin:$PATH\"", cuda_dir.display());

    Ok(cuda_dir)
}

/// Download CUDA installer
fn download_cuda(url: &str, dest: &Path) -> Result<()> {
    print!("Downloading CUDA Toolkit... ");
    io::stdout().flush()?;

    let response = reqwest::blocking::get(url).map_err(|e| Error::Http(e.to_string()))?;

    if !response.status().is_success() {
        return Err(Error::Http(format!(
            "Download failed with status: {}",
            response.status()
        )));
    }

    let total_size = response.content_length().unwrap_or(0);

    let mut file = fs::File::create(dest)?;
    let mut downloaded: u64 = 0;
    let mut stream = response;
    let mut last_print = 0.0;

    loop {
        let mut buffer = vec![0; 65536]; // 64KB buffer for faster download
        let bytes_read = io::Read::read(&mut stream, &mut buffer)?;
        if bytes_read == 0 {
            break;
        }

        io::Write::write_all(&mut file, &buffer[..bytes_read])?;
        downloaded += bytes_read as u64;

        if total_size > 0 {
            #[allow(clippy::cast_precision_loss)]
            let progress = (downloaded as f64 / total_size as f64) * 100.0;
            if progress - last_print >= 1.0 {
                print!("\rDownloading CUDA Toolkit... {progress:.0}%");
                io::stdout().flush()?;
                last_print = progress;
            }
        }
    }

    println!(
        "\rDownloading CUDA Toolkit... Done ({} MB)",
        downloaded / 1_000_000
    );
    Ok(())
}

/// Verify file checksum
fn verify_checksum(file_path: &Path, expected: &str) -> Result<()> {
    print!("Verifying checksum... ");
    io::stdout().flush()?;

    let mut file = fs::File::open(file_path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    let computed_hash = format!("{:x}", hasher.finalize());

    if computed_hash == expected {
        println!("OK");
        Ok(())
    } else {
        println!("FAILED");
        Err(Error::Sha256Mismatch {
            expected: expected.to_string(),
            actual: computed_hash,
        })
    }
}

/// Extract CUDA from the installer
fn extract_cuda(archive: &Path, dest: &Path) -> Result<()> {
    let filename = archive
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Error::Archive("Invalid archive path".into()))?;

    if filename.ends_with(".run") {
        extract_linux_runfile(archive, dest)
    } else if filename.ends_with(".exe") {
        extract_windows_exe(archive, dest)
    } else {
        Err(Error::Archive(format!(
            "Unsupported archive format: {filename}"
        )))
    }
}

/// Extract CUDA from Linux .run file
fn extract_linux_runfile(archive: &Path, dest: &Path) -> Result<()> {
    println!("Extracting CUDA Toolkit (this may take several minutes)...");

    // Create a temporary extraction directory
    let temp_dir = dest
        .parent()
        .ok_or_else(|| Error::Cuda("Invalid destination path".into()))?
        .join("tmp")
        .join("cuda_extract");

    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)?;
    }
    fs::create_dir_all(&temp_dir)?;

    // Make the .run file executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(archive)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(archive, perms)?;
    }

    // Extract using --extract flag
    // The .run file supports: --extract=<path> to extract without installing
    print!("Running CUDA installer extraction... ");
    io::stdout().flush()?;

    let status = Command::new("sh")
        .arg(archive)
        .arg("--silent")
        .arg("--toolkit")
        .arg(format!("--toolkitpath={}", dest.display()))
        .arg("--no-man-page")
        .arg("--no-opengl-libs")
        .arg("--no-drm")
        .status()
        .map_err(|e| Error::Cuda(format!("Failed to run CUDA installer: {e}")))?;

    if !status.success() {
        // If the full extraction fails, try the extract-only approach
        println!("Full extraction failed, trying alternative method...");

        let status = Command::new("sh")
            .arg(archive)
            .arg("--extract")
            .arg(&temp_dir)
            .status()
            .map_err(|e| Error::Cuda(format!("Failed to extract CUDA: {e}")))?;

        if !status.success() {
            return Err(Error::Cuda("CUDA extraction failed".into()));
        }

        // Copy only the components we need from the extracted files
        copy_cuda_components(&temp_dir, dest)?;

        // Clean up temp directory
        fs::remove_dir_all(&temp_dir)?;
    }

    println!("Done");
    Ok(())
}

/// Copy only the necessary CUDA components
fn copy_cuda_components(src: &Path, dest: &Path) -> Result<()> {
    print!("Copying CUDA components... ");
    io::stdout().flush()?;

    fs::create_dir_all(dest)?;

    // Components we need
    let components = ["cuda_nvcc", "cuda_cudart", "libcublas"];

    for component in &components {
        // Find the component directory (might be versioned)
        let entries = fs::read_dir(src)?;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(component) {
                copy_component(&entry.path(), dest)?;
            }
        }
    }

    println!("Done");
    Ok(())
}

/// Copy a CUDA component to the destination
fn copy_component(component_path: &Path, dest: &Path) -> Result<()> {
    // Each component has bin/, include/, lib64/ subdirectories
    let subdirs = ["bin", "include", "lib64", "lib"];

    for subdir in &subdirs {
        let src_subdir = component_path.join(subdir);
        if src_subdir.exists() {
            let dest_subdir = dest.join(subdir);
            fs::create_dir_all(&dest_subdir)?;
            copy_dir_contents(&src_subdir, &dest_subdir)?;
        }
    }

    Ok(())
}

/// Recursively copy directory contents
fn copy_dir_contents(src: &Path, dest: &Path) -> Result<()> {
    for entry in fs::read_dir(src)?.flatten() {
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());

        if src_path.is_dir() {
            fs::create_dir_all(&dest_path)?;
            copy_dir_contents(&src_path, &dest_path)?;
        } else {
            fs::copy(&src_path, &dest_path)?;
        }
    }
    Ok(())
}

/// Extract CUDA from Windows .exe installer
fn extract_windows_exe(archive: &Path, dest: &Path) -> Result<()> {
    println!("Extracting CUDA Toolkit...");

    let file = fs::File::open(archive)?;
    let len = file.metadata()?.len();
    let password = Password::empty();
    let mut reader =
        SevenZReader::new(file, len, password).map_err(|e| Error::Archive(e.to_string()))?;

    fs::create_dir_all(dest)?;

    // Extract only the components we need
    reader
        .for_each_entries(|entry, reader| {
            let entry_name = entry.name();

            // Filter for nvcc, cudart, and cublas components
            let dominated_components = ["nvcc", "cudart", "cublas", "cuda_runtime"];
            let dominated = dominated_components
                .iter()
                .any(|c| entry_name.to_lowercase().contains(c));

            if !dominated {
                return Ok(true); // Skip this entry
            }

            if entry.is_directory() {
                let dir_path = dest.join(entry_name);
                fs::create_dir_all(&dir_path).ok();
            } else {
                let file_path = dest.join(entry_name);
                if let Some(parent) = file_path.parent() {
                    fs::create_dir_all(parent).ok();
                }
                let mut output = fs::File::create(&file_path)?;
                io::copy(reader, &mut output)?;
            }
            Ok(true)
        })
        .map_err(|e| Error::Archive(e.to_string()))?;

    println!("Done");
    Ok(())
}

/// Uninstall CUDA from `~/.pecos/deps/cuda/`
///
/// # Errors
/// Returns an error if:
/// - Home directory cannot be determined
/// - Directory removal fails
pub fn uninstall_cuda() -> Result<()> {
    let cuda_dir = get_pecos_cuda_dir()?;

    if !cuda_dir.exists() {
        println!("CUDA is not installed in ~/.pecos/deps/cuda/");
        return Ok(());
    }

    println!("Removing CUDA installation at: {}", cuda_dir.display());
    fs::remove_dir_all(&cuda_dir)?;
    println!("CUDA uninstalled successfully");

    Ok(())
}
