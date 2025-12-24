//! LLVM 14.0.6 installation functionality

#![allow(clippy::case_sensitive_file_extension_comparisons)]

use crate::errors::{Error, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Known SHA256 checksums for LLVM 14.0.6 downloads
const LLVM_CHECKSUMS: &[(&str, &str)] = &[
    (
        "clang+llvm-14.0.6-x86_64-apple-darwin.tar.xz",
        "e6cc6b8279661fd4452c2847cb8e55ce1e54e1faf4ab497b37c85ffdb6685e7c",
    ),
    (
        "clang+llvm-14.0.6-arm64-apple-darwin22.3.0.tar.xz",
        "82f4f7607a16c9aaf7314b945bde6a4639836ec9d2b474ebb3a31dee33e3c15a",
    ),
    (
        "clang+llvm-14.0.6-x86_64-linux-gnu-rhel-8.4.tar.xz",
        "7412026be8bb8f6b4c25ef58c7a1f78ed5ea039d94f0fa633a386de9c60a6942",
    ),
    (
        "clang+llvm-14.0.6-aarch64-linux-gnu.tar.xz",
        "1a81fda984f5e607584916fdf69cf41e5385b219b983544d2c1a14950d5a65cf",
    ),
    (
        "LLVM-14.0.6-win64.7z",
        "611e7a39363a2b63267d012a05f83ea9ce2b432a448890459c9412233327ac11",
    ),
];

/// Install LLVM 14.0.6 to `~/.pecos/llvm/`
///
/// # Arguments
/// * `force` - Force reinstall even if already present
/// * `no_configure` - Skip automatic configuration after installation
///
/// # Errors
///
/// Returns an error if installation fails
pub fn install_llvm(force: bool, no_configure: bool) -> Result<PathBuf> {
    let llvm_dir = dirs::home_dir()
        .ok_or_else(|| Error::HomeDir("Could not determine home directory".into()))?
        .join(".pecos")
        .join("llvm");

    // Check if already installed
    if !force && llvm_dir.exists() && is_valid_installation(&llvm_dir) {
        return Err(Error::Llvm(
            "LLVM is already installed. Use --force to reinstall.".into(),
        ));
    }

    // Remove existing if force
    if force && llvm_dir.exists() {
        println!("Removing existing LLVM installation...");
        fs::remove_dir_all(&llvm_dir)?;
    }

    println!("Installing LLVM 14.0.6...");
    println!("This will download ~400MB and may take 5-10 minutes.");
    println!();

    let (url, archive_name) = get_download_url()?;

    // Create parent directory
    if let Some(parent) = llvm_dir.parent() {
        fs::create_dir_all(parent)?;
    }

    // Download to temp directory
    let temp_base = llvm_dir.parent().unwrap_or(&llvm_dir).join("tmp");
    let temp_dir = temp_base.join("llvm");
    fs::create_dir_all(&temp_dir)?;
    let archive_path = temp_dir.join(&archive_name);

    // Download and verify
    download_and_verify_with_retry(&url, &archive_path, &archive_name)?;

    // Extract
    extract_llvm(&archive_path, &llvm_dir)?;

    // Cleanup
    fs::remove_dir_all(&temp_dir)?;

    // Apply platform-specific fixes
    apply_platform_fixes(&llvm_dir)?;

    // Verify
    if !is_valid_installation(&llvm_dir) {
        return Err(Error::Llvm(
            "Installation completed but verification failed".into(),
        ));
    }

    verify_llvm_runtime(&llvm_dir)?;

    println!();
    println!("Installation complete!");
    println!("LLVM 14.0.6 installed to: {}", llvm_dir.display());

    if no_configure {
        println!();
        println!("Skipping automatic configuration (--no-configure specified).");
        println!();
        println!("To configure PECOS, run:");
        println!("  pecos-deps llvm configure");
    } else {
        println!();
        println!("Configuring PECOS to use this LLVM installation...");
        match super::config::auto_configure_llvm(None) {
            Ok(configured_path) => {
                println!("Updated .cargo/config.toml with LLVM configuration");
                println!("Configured LLVM path: {}", configured_path.display());
            }
            Err(e) => {
                eprintln!("Warning: Could not auto-configure LLVM: {e}");
                println!();
                println!("Please run configuration manually:");
                println!("  pecos-deps llvm configure");
            }
        }
    }

    Ok(llvm_dir)
}

fn get_download_url() -> Result<(String, String)> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    match os {
        "macos" => {
            if arch == "aarch64" {
                Ok((
                    "https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.6/clang+llvm-14.0.6-arm64-apple-darwin22.3.0.tar.xz".to_string(),
                    "clang+llvm-14.0.6-arm64-apple-darwin22.3.0.tar.xz".to_string(),
                ))
            } else {
                Ok((
                    "https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.6/clang+llvm-14.0.6-x86_64-apple-darwin.tar.xz".to_string(),
                    "clang+llvm-14.0.6-x86_64-apple-darwin.tar.xz".to_string(),
                ))
            }
        }
        "linux" => {
            if arch == "x86_64" {
                Ok((
                    "https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.6/clang+llvm-14.0.6-x86_64-linux-gnu-rhel-8.4.tar.xz".to_string(),
                    "clang+llvm-14.0.6-x86_64-linux-gnu-rhel-8.4.tar.xz".to_string(),
                ))
            } else if arch == "aarch64" {
                Ok((
                    "https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.6/clang+llvm-14.0.6-aarch64-linux-gnu.tar.xz".to_string(),
                    "clang+llvm-14.0.6-aarch64-linux-gnu.tar.xz".to_string(),
                ))
            } else {
                Err(Error::Llvm(format!("Unsupported Linux architecture: {arch}")))
            }
        }
        "windows" => Ok((
            "https://github.com/PLC-lang/llvm-package-windows/releases/download/v14.0.6/LLVM-14.0.6-win64.7z".to_string(),
            "LLVM-14.0.6-win64.7z".to_string(),
        )),
        _ => Err(Error::Llvm(format!("Unsupported operating system: {os}"))),
    }
}

fn download_and_verify_with_retry(url: &str, dest: &PathBuf, archive_name: &str) -> Result<()> {
    const MAX_RETRIES: u32 = 5;
    const BASE_DELAY_SECS: u64 = 10;

    for attempt in 1..=MAX_RETRIES {
        if attempt > 1 {
            // Exponential backoff: 10s, 20s, 40s, 80s
            let delay_secs = BASE_DELAY_SECS * (1 << (attempt - 2));
            println!();
            println!("Retry attempt {attempt}/{MAX_RETRIES} (waiting {delay_secs}s)...");
            std::thread::sleep(std::time::Duration::from_secs(delay_secs));
        }

        let _ = fs::remove_file(dest);

        if let Err(e) = download_llvm(url, dest) {
            if attempt < MAX_RETRIES {
                eprintln!("Download error: {e}");
                continue;
            }
            return Err(e);
        }

        // Check for empty downloads (CDN/rate limit issues)
        let file_size = fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
        if file_size == 0 {
            if attempt < MAX_RETRIES {
                eprintln!("Download returned empty file (possible CDN issue)");
                continue;
            }
            return Err(Error::Llvm(
                "Download returned empty file after all retries".into(),
            ));
        }

        match verify_checksum(dest, archive_name) {
            Ok(()) => return Ok(()),
            Err(e) => {
                if attempt < MAX_RETRIES {
                    eprintln!();
                    eprintln!("Checksum verification failed. Retrying...");
                    let _ = fs::remove_file(dest);
                    continue;
                }
                return Err(e);
            }
        }
    }

    Err(Error::Llvm(
        "Download and verification failed after all retries".into(),
    ))
}

fn download_llvm(url: &str, dest: &PathBuf) -> Result<()> {
    print!("Downloading LLVM... ");
    io::Write::flush(&mut io::stdout())?;

    let response = reqwest::blocking::get(url).map_err(|e| Error::Http(e.to_string()))?;
    let total_size = response.content_length().unwrap_or(0);

    let mut file = fs::File::create(dest)?;
    let mut downloaded: u64 = 0;
    let mut stream = response;
    let mut last_print = 0.0;

    loop {
        let mut buffer = vec![0; 8192];
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
                print!("\rDownloading LLVM... {progress:.0}%");
                io::Write::flush(&mut io::stdout())?;
                last_print = progress;
            }
        }
    }

    println!("\rDownloading LLVM... Done ({} MB)", downloaded / 1_000_000);
    Ok(())
}

fn verify_checksum(file_path: &PathBuf, archive_name: &str) -> Result<()> {
    print!("Verifying checksum... ");
    io::Write::flush(&mut io::stdout())?;

    let mut file = fs::File::open(file_path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    let computed_hash = format!("{:x}", hasher.finalize());

    let expected_hash = LLVM_CHECKSUMS
        .iter()
        .find(|(name, _)| *name == archive_name)
        .map(|(_, hash)| *hash);

    match expected_hash {
        Some(expected) if !expected.is_empty() => {
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
        _ => {
            println!("Skipped (checksum not available)");
            Ok(())
        }
    }
}

fn extract_llvm(archive: &PathBuf, dest: &PathBuf) -> Result<()> {
    print!("Extracting LLVM... ");
    io::Write::flush(&mut io::stdout())?;

    let file_name = archive
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Error::Archive("Could not determine archive name".into()))?;

    if file_name.ends_with(".tar.xz") {
        extract_tar_xz(archive, dest)?;
    } else if file_name.ends_with(".7z") {
        extract_7z(archive, dest)?;
    } else {
        return Err(Error::Archive(format!(
            "Unsupported archive format: {file_name}"
        )));
    }

    println!("Done");
    Ok(())
}

fn extract_tar_xz(archive: &PathBuf, dest: &PathBuf) -> Result<()> {
    use tar::Archive;
    use xz2::read::XzDecoder;

    let file = fs::File::open(archive)?;
    let decompressor = XzDecoder::new(file);
    let mut tar_archive = Archive::new(decompressor);

    let extract_to = dest
        .parent()
        .ok_or_else(|| Error::Archive("Invalid destination path".into()))?;
    tar_archive.unpack(extract_to)?;

    // Find and rename extracted directory
    let archive_name = archive.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let archive_path_buf = PathBuf::from(archive_name);
    let base_name = archive_path_buf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(archive_name);
    let extracted_dir = extract_to.join(base_name);

    if extracted_dir.exists() && !dest.exists() {
        fs::rename(&extracted_dir, dest)?;
    }

    Ok(())
}

fn extract_7z(archive: &PathBuf, dest: &PathBuf) -> Result<()> {
    use sevenz_rust::{Password, SevenZReader};

    let file = fs::File::open(archive)?;
    let len = file.metadata()?.len();
    let password = Password::empty();
    let mut reader =
        SevenZReader::new(file, len, password).map_err(|e| Error::Archive(e.to_string()))?;

    // Windows LLVM archives have flat structure (bin/, lib/, etc. at root)
    // Extract directly to destination
    fs::create_dir_all(dest)?;

    reader
        .for_each_entries(|entry, reader| {
            let entry_name = entry.name();

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

    Ok(())
}

/// Validate that a path contains a complete LLVM 14 installation
#[must_use]
pub fn is_valid_installation(path: &Path) -> bool {
    let exe_ext = if cfg!(windows) { ".exe" } else { "" };

    let critical_executables = [
        format!("bin/llvm-config{exe_ext}"),
        format!("bin/clang{exe_ext}"),
    ];

    for exe in &critical_executables {
        if !path.join(exe).exists() {
            return false;
        }
    }

    true
}

fn verify_llvm_runtime(llvm_dir: &Path) -> Result<()> {
    print!("Verifying LLVM runtime... ");
    io::Write::flush(&mut io::stdout())?;

    let llvm_config = if cfg!(windows) {
        llvm_dir.join("bin").join("llvm-config.exe")
    } else {
        llvm_dir.join("bin").join("llvm-config")
    };

    let output = std::process::Command::new(&llvm_config)
        .arg("--version")
        .output()
        .map_err(|e| Error::Llvm(format!("Failed to execute llvm-config: {e}")))?;

    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if version.starts_with("14.0") {
            println!("OK (version {version})");
            Ok(())
        } else {
            println!("FAILED");
            Err(Error::Llvm(format!("Unexpected LLVM version: {version}")))
        }
    } else {
        println!("FAILED");
        Err(Error::Llvm(
            "llvm-config exited with non-zero status".into(),
        ))
    }
}

#[cfg(target_os = "macos")]
fn apply_platform_fixes(llvm_dir: &Path) -> Result<()> {
    use std::process::Command;

    print!("Applying macOS platform fixes... ");
    io::Write::flush(&mut io::stdout())?;

    let lib_dir = llvm_dir.join("lib");
    let libunwind = lib_dir.join("libunwind.1.0.dylib");

    if !libunwind.exists() {
        println!("Skipped (libunwind not found)");
        return Ok(());
    }

    let new_install_name = lib_dir.join("libunwind.1.dylib");

    let status = Command::new("install_name_tool")
        .arg("-id")
        .arg(&new_install_name)
        .arg(&libunwind)
        .status()?;

    if !status.success() {
        println!("FAILED");
        return Err(Error::Llvm("install_name_tool failed".into()));
    }

    println!("OK");
    Ok(())
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::unnecessary_wraps)]
fn apply_platform_fixes(_llvm_dir: &Path) -> Result<()> {
    Ok(())
}
