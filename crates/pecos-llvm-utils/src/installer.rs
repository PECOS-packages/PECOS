//! LLVM 14.0.6 installation functionality
//!
//! Downloads and extracts LLVM 14.0.6 pre-built binaries to a project-local directory.

use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Known SHA256 checksums for LLVM 14.0.6 downloads
/// Format: (filename, `sha256_hash`)
///
/// To compute checksums for new files:
///   sha256sum <file>  # Linux/macOS
///   Get-FileHash -Algorithm SHA256 <file>  # Windows `PowerShell`
const LLVM_CHECKSUMS: &[(&str, &str)] = &[
    // macOS Intel
    (
        "clang+llvm-14.0.6-x86_64-apple-darwin.tar.xz",
        "e6cc6b8279661fd4452c2847cb8e55ce1e54e1faf4ab497b37c85ffdb6685e7c",
    ),
    // macOS Apple Silicon
    (
        "clang+llvm-14.0.6-arm64-apple-darwin22.3.0.tar.xz",
        "82f4f7607a16c9aaf7314b945bde6a4639836ec9d2b474ebb3a31dee33e3c15a",
    ),
    // Linux x86_64
    (
        "clang+llvm-14.0.6-x86_64-linux-gnu-rhel-8.4.tar.xz",
        "7412026be8bb8f6b4c25ef58c7a1f78ed5ea039d94f0fa633a386de9c60a6942",
    ),
    // Linux aarch64
    (
        "clang+llvm-14.0.6-aarch64-linux-gnu.tar.xz",
        "7412026be8bb8f6b4c25ef58c7a1f78ed5ea039d94f0fa633a386de9c60a6942",
    ),
    // Windows (from PLC-lang/llvm-package-windows)
    (
        "LLVM-14.0.6-win64.7z",
        "611e7a39363a2b63267d012a05f83ea9ce2b432a448890459c9412233327ac11",
    ),
];

/// Install LLVM 14.0.6 to ~/.pecos/llvm/
///
/// Downloads and installs LLVM 14.0.6 pre-built binaries to a PECOS-managed
/// directory at ~/.pecos/llvm/. This ensures a clean, isolated installation
/// that PECOS can safely modify (e.g., fixing dylib references on macOS).
///
/// After installation, run `pecos-llvm configure` to update .cargo/config.toml,
/// or set the `LLVM_SYS_140_PREFIX` environment variable to `~/.pecos/llvm` manually.
///
/// # Arguments
/// * `force` - Force reinstall even if already present
/// * `no_configure` - Skip automatic configuration after installation
///
/// # Errors
/// Returns an error if:
/// - LLVM is already installed and `force` is false
/// - The download or extraction fails
/// - Installation verification fails
/// - Platform fixes fail (e.g., `install_name_tool` on macOS)
///
/// # Returns
/// Path to the installed LLVM directory (~/.pecos/llvm/)
pub fn install_llvm(
    force: bool,
    no_configure: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // PECOS-managed installation: ~/.pecos/llvm
    let llvm_dir = dirs::home_dir()
        .ok_or("Could not determine home directory")?
        .join(".pecos")
        .join("llvm");

    // Check if already installed
    if !force && llvm_dir.exists() && is_valid_installation(&llvm_dir) {
        return Err("LLVM is already installed. Use --force to reinstall.".into());
    }

    // If force is specified and directory exists, remove it first
    if force && llvm_dir.exists() {
        println!("Removing existing LLVM installation...");
        fs::remove_dir_all(&llvm_dir)?;
    }

    println!("Installing LLVM 14.0.6...");
    println!("This will download ~400MB and may take 5-10 minutes.");
    println!();

    // Determine platform and download URL
    let (url, archive_name) = get_download_url()?;

    // Create parent directory if it doesn't exist
    if let Some(parent) = llvm_dir.parent() {
        fs::create_dir_all(parent)?;
    }

    // Download to temp directory (use llvm subdirectory to avoid conflicts)
    let temp_base = llvm_dir.parent().unwrap_or(&llvm_dir).join("tmp");
    let temp_dir = temp_base.join("llvm");
    fs::create_dir_all(&temp_dir)?;
    let archive_path = temp_dir.join(&archive_name);
    download_llvm(&url, &archive_path)?;

    // Verify checksum
    verify_checksum(&archive_path, &archive_name)?;

    // Extract
    extract_llvm(&archive_path, &llvm_dir)?;

    // Cleanup LLVM temp directory only (not entire tmp directory)
    fs::remove_dir_all(&temp_dir)?;

    // Apply platform-specific fixes (e.g., fix libunwind on macOS)
    apply_platform_fixes(&llvm_dir)?;

    // Verify installation files
    if !is_valid_installation(&llvm_dir) {
        return Err("Installation completed but file verification failed".into());
    }

    // Verify runtime functionality
    verify_llvm_runtime(&llvm_dir)?;

    println!();
    println!("Installation complete!");
    println!("LLVM 14.0.6 installed to: {}", llvm_dir.display());

    // Configure LLVM (unless --no-configure is specified)
    if no_configure {
        println!();
        println!("Skipping automatic configuration (--no-configure specified).");
        println!();
        println!("To configure PECOS, run:");
        println!("  pecos-llvm configure");
        println!();
        println!("Or set the environment variable manually:");
        println!("  export LLVM_SYS_140_PREFIX=\"{}\"", llvm_dir.display());
    } else {
        println!();
        println!("Configuring PECOS to use this LLVM installation...");
        match crate::auto_configure_llvm(None) {
            Ok(configured_path) => {
                println!("Updated .cargo/config.toml with LLVM configuration");
                println!("Configured LLVM path: {}", configured_path.display());
                println!();
                println!("You can now build PECOS:");
                println!("  cargo build");
            }
            Err(e) => {
                eprintln!("Warning: Could not auto-configure LLVM: {e}");
                println!();
                println!("Please run configuration manually:");
                println!("  pecos-llvm configure");
            }
        }
    }

    Ok(llvm_dir)
}

fn get_download_url() -> Result<(String, String), Box<dyn std::error::Error>> {
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
                Err(format!("Unsupported Linux architecture: {arch}").into())
            }
        }
        "windows" => {
            Ok((
                "https://github.com/PLC-lang/llvm-package-windows/releases/download/v14.0.6/LLVM-14.0.6-win64.7z".to_string(),
                "LLVM-14.0.6-win64.7z".to_string(),
            ))
        }
        _ => Err(format!("Unsupported operating system: {os}").into()),
    }
}

fn download_llvm(url: &str, dest: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    print!("Downloading LLVM... ");
    io::Write::flush(&mut io::stdout())?;

    let response = reqwest::blocking::get(url)?;
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
            // Precision loss is acceptable for progress display
            #[allow(clippy::cast_precision_loss)]
            let progress = (downloaded as f64 / total_size as f64) * 100.0;
            // Only update display every 1%
            if progress - last_print >= 1.0 {
                print!("\rDownloading LLVM... {progress:.0}%");
                io::Write::flush(&mut io::stdout())?;
                last_print = progress;
            }
        }
    }

    println!("\rDownloading LLVM... Done");
    Ok(())
}

fn verify_checksum(
    file_path: &PathBuf,
    archive_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    print!("Verifying checksum... ");
    io::Write::flush(&mut io::stdout())?;

    // Compute SHA256 of downloaded file
    let mut file = fs::File::open(file_path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    let computed_hash = format!("{:x}", hasher.finalize());

    // Look up expected checksum
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
                eprintln!();
                eprintln!("═══════════════════════════════════════════════════════════════");
                eprintln!("CHECKSUM VERIFICATION FAILED");
                eprintln!("═══════════════════════════════════════════════════════════════");
                eprintln!();
                eprintln!("File: {archive_name}");
                eprintln!("Expected: {expected}");
                eprintln!("Computed: {computed_hash}");
                eprintln!();
                eprintln!("This could indicate:");
                eprintln!("  - A corrupted download");
                eprintln!("  - A compromised source");
                eprintln!("  - A network error during download");
                eprintln!();
                eprintln!("Please try again or download manually from:");
                eprintln!("  https://github.com/llvm/llvm-project/releases/tag/llvmorg-14.0.6");
                eprintln!("═══════════════════════════════════════════════════════════════");
                Err("Checksum verification failed".into())
            }
        }
        Some(_) | None => {
            // Checksum not available - display computed hash
            println!("Skipped (checksum not available)");
            println!();
            println!("  WARNING: Computed SHA256: {computed_hash}");
            println!("  Please verify this matches the official checksum for security.");
            println!();
            Ok(())
        }
    }
}

fn extract_llvm(archive: &PathBuf, dest: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    print!("Extracting LLVM... ");
    io::Write::flush(&mut io::stdout())?;

    // Determine archive type using Path::extension() for case-insensitive comparison
    let file_name = archive
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("Could not determine archive name")?;

    // Check for .tar.xz (compound extension)
    if file_name.ends_with(".tar.xz") || file_name.ends_with(".tar.XZ") {
        extract_tar_xz(archive, dest)?;
    } else if std::path::Path::new(file_name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("7z"))
    {
        extract_7z(archive, dest)?;
    } else {
        return Err(format!("Unsupported archive format: {file_name}").into());
    }

    println!("Done");
    Ok(())
}

fn extract_tar_xz(archive: &PathBuf, dest: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    use tar::Archive;
    use xz2::read::XzDecoder;

    // Open the .tar.xz file
    let file = fs::File::open(archive)?;
    let decompressor = XzDecoder::new(file);
    let mut tar_archive = Archive::new(decompressor);

    // Extract to parent directory first
    let extract_to = dest.parent().ok_or("Invalid destination path")?;
    tar_archive.unpack(extract_to)?;

    // The archive extracts to a directory like clang+llvm-14.0.6-...
    // We need to determine the extracted directory name from the archive filename
    let archive_name = archive
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("Could not determine archive name")?;

    // For .tar.xz, we need to strip the .tar part too
    let archive_path_buf = PathBuf::from(archive_name);
    let base_name = if let Some(stem) = archive_path_buf.file_stem() {
        stem.to_str().ok_or("Invalid archive name")?
    } else {
        archive_name
    };

    let extracted_dir = extract_to.join(base_name);

    // If dest doesn't exist, rename extracted_dir to dest
    if dest.exists() {
        // Move contents
        for entry in fs::read_dir(&extracted_dir)? {
            let entry = entry?;
            let dest_path = dest.join(entry.file_name());
            fs::rename(entry.path(), dest_path)?;
        }
        fs::remove_dir(&extracted_dir)?;
    } else {
        fs::rename(&extracted_dir, dest)?;
    }

    Ok(())
}

fn extract_7z(archive: &PathBuf, dest: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    use sevenz_rust::{Password, SevenZReader};

    // Open the .7z file
    let file = fs::File::open(archive)?;
    let len = file.metadata()?.len();
    let password = Password::empty();
    let mut reader = SevenZReader::new(file, len, password)?;

    // Extract to parent directory first
    let extract_to = dest.parent().ok_or("Invalid destination path")?;
    fs::create_dir_all(extract_to)?;

    // Extract all files
    reader.for_each_entries(|entry, reader| {
        if entry.is_directory() {
            let dir_path = extract_to.join(entry.name());
            fs::create_dir_all(&dir_path).ok();
        } else {
            let file_path = extract_to.join(entry.name());
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent).ok();
            }
            let mut output = fs::File::create(&file_path)?;
            io::copy(reader, &mut output)?;
        }
        Ok(true) // Continue extracting
    })?;

    // Check if LLVM was extracted directly to extract_to (no wrapper directory)
    // This is the case for some Windows 7z archives
    let llvm_config = if cfg!(windows) {
        extract_to.join("bin").join("llvm-config.exe")
    } else {
        extract_to.join("bin").join("llvm-config")
    };

    if llvm_config.exists() {
        // LLVM was extracted directly to extract_to, move it to dest
        fs::create_dir_all(dest)?;
        for entry in fs::read_dir(extract_to)? {
            let entry = entry?;
            let entry_path = entry.path();
            // Skip the dest directory itself and the tmp directory
            if entry_path == *dest || entry.file_name() == "tmp" {
                continue;
            }
            let dest_path = dest.join(entry.file_name());
            fs::rename(entry_path, dest_path)?;
        }
    } else {
        // The archive extracts to a directory like LLVM-14.0.6-win64
        // Find the extracted directory
        let mut extracted_dir = None;
        let mut found_dirs = Vec::new();

        for entry in fs::read_dir(extract_to)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir()
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
            {
                found_dirs.push(name.to_string());
                // Case-insensitive search for "LLVM" in directory name
                if name.to_uppercase().contains("LLVM") {
                    extracted_dir = Some(path);
                    break;
                }
            }
        }

        // If we found a subdirectory with "LLVM" in the name, use it
        if let Some(extracted_dir) = extracted_dir {
            // If dest doesn't exist, rename extracted_dir to dest
            if dest.exists() {
                // Move contents
                for entry in fs::read_dir(&extracted_dir)? {
                    let entry = entry?;
                    let dest_path = dest.join(entry.file_name());
                    fs::rename(entry.path(), dest_path)?;
                }
                fs::remove_dir(&extracted_dir)?;
            } else {
                fs::rename(&extracted_dir, dest)?;
            }
        } else {
            // No subdirectory found with "LLVM" in name
            // Check if there's only one directory - it might be the LLVM directory with a different name
            if found_dirs.len() == 1 {
                // Assume this single directory is the LLVM installation
                let single_dir = extract_to.join(&found_dirs[0]);
                if dest.exists() {
                    // Move contents
                    for entry in fs::read_dir(&single_dir)? {
                        let entry = entry?;
                        let dest_path = dest.join(entry.file_name());
                        fs::rename(entry.path(), dest_path)?;
                    }
                    fs::remove_dir(&single_dir)?;
                } else {
                    fs::rename(&single_dir, dest)?;
                }
            } else {
                return Err(format!(
                    "Could not find extracted LLVM directory. Expected directory with 'LLVM' in name or bin/llvm-config. Found directories: {found_dirs:?}"
                )
                .into());
            }
        }
    }

    Ok(())
}

/// Validate that a path contains a complete LLVM 14 installation
///
/// Checks for critical executables, libraries, and header files.
///
/// # Arguments
/// * `path` - Path to the LLVM installation directory
///
/// # Returns
/// `true` if all critical components are present, `false` otherwise
#[must_use]
pub fn is_valid_installation(path: &Path) -> bool {
    // Check critical executable files
    let exe_ext = if cfg!(windows) { ".exe" } else { "" };

    let critical_executables = [
        format!("bin/llvm-config{exe_ext}"),
        format!("bin/clang{exe_ext}"),
        format!("bin/llvm-ar{exe_ext}"),
        format!("bin/llvm-as{exe_ext}"),
    ];

    for exe in &critical_executables {
        if !path.join(exe).exists() {
            eprintln!("Validation failed: Missing critical executable: {exe}");
            return false;
        }
    }

    // Check critical library files
    let lib_ext = if cfg!(windows) { "lib" } else { "a" };

    // Check for at least one core LLVM library (different naming on different platforms)
    let has_llvm_lib = if cfg!(windows) {
        // Windows: check for LLVM-C.lib, LTO.lib, or individual component libraries
        path.join("lib").join("LLVM-C.lib").exists()
            || path.join("lib").join("LTO.lib").exists()
            || path.join("lib").join("LLVMCore.lib").exists()
    } else {
        // Unix: check for monolithic libraries or individual component libraries
        path.join("lib")
            .join(format!("libLLVM-14.{lib_ext}"))
            .exists()
            || path.join("lib").join(format!("libLLVM.{lib_ext}")).exists()
            || path
                .join("lib")
                .join(format!("libLLVMCore.{lib_ext}"))
                .exists()
    };

    if !has_llvm_lib {
        eprintln!("Validation failed: Missing LLVM core libraries in lib/");
        return false;
    }

    // Check critical header files
    let critical_headers = [
        "include/llvm-c/Core.h",
        "include/llvm/IR/Module.h",
        "include/llvm/Support/CommandLine.h",
    ];

    for header in &critical_headers {
        if !path.join(header).exists() {
            eprintln!("Validation failed: Missing critical header: {header}");
            return false;
        }
    }

    true
}

/// Verify that LLVM runtime is functional by executing llvm-config
///
/// # Arguments
/// * `llvm_dir` - Path to the LLVM installation directory
///
/// # Returns
/// * `Ok(())` if llvm-config executes successfully and reports version 14.0.x
///
/// # Errors
/// Returns an error if:
/// * IO operations fail (stdout flush)
/// * llvm-config fails to execute
/// * llvm-config reports a version other than 14.0.x
pub fn verify_llvm_runtime(llvm_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    print!("Verifying LLVM runtime... ");
    io::Write::flush(&mut io::stdout())?;

    let llvm_config = if cfg!(windows) {
        llvm_dir.join("bin").join("llvm-config.exe")
    } else {
        llvm_dir.join("bin").join("llvm-config")
    };

    // Try to run llvm-config --version
    let output = std::process::Command::new(&llvm_config)
        .arg("--version")
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            let version = version.trim();

            // Check that version starts with 14.0
            if version.starts_with("14.0") {
                println!("OK (version {version})");
                Ok(())
            } else {
                println!("FAILED");
                Err(format!("Unexpected LLVM version: {version} (expected 14.0.x)").into())
            }
        }
        Ok(_) => {
            println!("FAILED");
            Err("llvm-config exited with non-zero status".into())
        }
        Err(e) => {
            println!("FAILED");
            Err(format!("Failed to execute llvm-config: {e}").into())
        }
    }
}

/// Apply platform-specific fixes to the LLVM installation
///
/// On macOS, fixes the libunwind dylib install name to use an absolute path
/// instead of @rpath, which prevents runtime linking issues.
///
/// # Arguments
/// * `llvm_dir` - Path to the LLVM installation directory
///
/// # Errors
/// Returns an error if `install_name_tool` fails to execute
#[cfg(target_os = "macos")]
fn apply_platform_fixes(llvm_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;

    print!("Applying macOS platform fixes... ");
    io::Write::flush(&mut io::stdout())?;

    let lib_dir = llvm_dir.join("lib");
    let libunwind = lib_dir.join("libunwind.1.0.dylib");

    if !libunwind.exists() {
        println!("Skipped (libunwind not found)");
        return Ok(());
    }

    // Fix libunwind's install name from @rpath to absolute path
    // This prevents "Library not loaded: @rpath/libunwind.1.dylib" errors
    let new_install_name = lib_dir.join("libunwind.1.dylib");

    let status = Command::new("install_name_tool")
        .arg("-id")
        .arg(&new_install_name)
        .arg(&libunwind)
        .status()?;

    if !status.success() {
        println!("FAILED");
        return Err("install_name_tool failed to fix libunwind".into());
    }

    println!("OK");
    Ok(())
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::unnecessary_wraps)]
fn apply_platform_fixes(_llvm_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // No platform fixes needed on non-macOS platforms
    Ok(())
}
