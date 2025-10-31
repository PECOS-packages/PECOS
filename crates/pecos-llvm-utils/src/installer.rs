//! LLVM 14.0.6 installation functionality
//!
//! Downloads and extracts LLVM 14.0.6 pre-built binaries to a project-local directory.

use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Installation location type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallLocation {
    /// PECOS-managed installation at ~/.pecos/llvm/ (default)
    /// Uses .cargo/config.toml for configuration
    PecosManaged,
    /// System-wide installation at standard location (requires admin/sudo)
    /// Sets `LLVM_SYS_140_PREFIX` environment variable
    System,
    /// User-level installation at standard location
    /// Sets `LLVM_SYS_140_PREFIX` environment variable
    User,
}

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

/// Install LLVM 14.0.6 to specified location
///
/// Installs to location based on `InstallLocation`:
///
/// **`PecosManaged`** (default):
/// - All platforms: ~/.pecos/llvm
/// - Configures via .cargo/config.toml
///
/// **System** (--system flag):
/// - Windows: C:\Program Files\LLVM-14 (requires admin)
/// - Unix: /usr/local/LLVM-14 (requires sudo)
/// - Sets `LLVM_SYS_140_PREFIX` environment variable permanently
///
/// **User** (--user flag):
/// - Windows: %LOCALAPPDATA%\Programs\LLVM-14
/// - macOS: ~/Library/Application Support/LLVM-14
/// - Linux: ~/.local/LLVM-14
/// - Sets `LLVM_SYS_140_PREFIX` environment variable permanently
///
/// # Arguments
/// * `force` - Force reinstall even if already present
/// * `no_configure` - Skip automatic configuration after installation
/// * `location` - Where to install LLVM
///
/// # Errors
/// Returns an error if:
/// - LLVM is already installed and `force` is false
/// - The download or extraction fails
/// - Platform-specific fixes fail
/// - Installation verification fails
/// - Setting environment variable fails (System/User installs)
///
/// # Returns
/// Path to the installed LLVM directory
pub fn install_llvm(
    force: bool,
    no_configure: bool,
    location: InstallLocation,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let llvm_dir = get_install_location(location)?;

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

    // Apply platform-specific fixes
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

    // Configure LLVM based on installation location (unless --no-configure is specified)
    if no_configure {
        println!();
        println!("Skipping automatic configuration (--no-configure specified).");
        match location {
            InstallLocation::PecosManaged => {
                println!();
                println!("To configure PECOS, run:");
                println!("  cargo run -p pecos-llvm-utils --bin pecos-llvm -- configure");
            }
            InstallLocation::System | InstallLocation::User => {
                println!();
                println!("To make this installation permanent, set the environment variable:");
                println!("  LLVM_SYS_140_PREFIX={}", llvm_dir.display());
            }
        }
    } else {
        println!();
        match location {
            InstallLocation::PecosManaged => {
                // PECOS-managed: use .cargo/config.toml
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
                        println!("  cargo run -p pecos-llvm-utils --bin pecos-llvm -- configure");
                    }
                }
            }
            InstallLocation::System | InstallLocation::User => {
                // System/User: set LLVM_SYS_140_PREFIX environment variable
                println!("Setting LLVM_SYS_140_PREFIX environment variable...");
                match set_llvm_env_var(&llvm_dir, location == InstallLocation::System) {
                    Ok(()) => {
                        println!("Environment variable set successfully!");
                        println!("LLVM_SYS_140_PREFIX={}", llvm_dir.display());
                        println!();
                        println!("You can now build PECOS:");
                        println!("  cargo build");
                        println!();
                        if location == InstallLocation::System {
                            println!("Note: You may need to restart your terminal or system for");
                            println!("system-wide environment changes to take effect.");
                        } else {
                            println!("Note: You may need to restart your terminal for the");
                            println!("environment variable to take effect in new sessions.");
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: Could not set environment variable: {e}");
                        println!();
                        println!("Please set it manually:");
                        #[cfg(target_os = "windows")]
                        println!("  setx LLVM_SYS_140_PREFIX \"{}\"", llvm_dir.display());
                        #[cfg(not(target_os = "windows"))]
                        println!("  export LLVM_SYS_140_PREFIX=\"{}\"", llvm_dir.display());
                    }
                }
            }
        }
    }

    Ok(llvm_dir)
}

/// Set `LLVM_SYS_140_PREFIX` environment variable permanently
///
/// # Arguments
/// * `llvm_path` - Path to the LLVM installation
/// * `system_wide` - If true, set system-wide; if false, set for current user
///
/// # Errors
/// Returns an error if setting the environment variable fails
fn set_llvm_env_var(llvm_path: &Path, system_wide: bool) -> Result<(), Box<dyn std::error::Error>> {
    let path_str = llvm_path.to_string_lossy();

    #[cfg(target_os = "windows")]
    {
        use std::process::Command;

        if system_wide {
            // System-wide on Windows requires admin rights and registry manipulation
            // For now, we'll use setx which sets user-level, and warn about system-level
            eprintln!(
                "Warning: System-wide environment variables on Windows require administrator"
            );
            eprintln!("privileges and registry modification. Setting user-level variable instead.");
            eprintln!();
            eprintln!("To set system-wide, run as administrator:");
            eprintln!("  setx /M LLVM_SYS_140_PREFIX \"{path_str}\"");
            eprintln!();
        }

        // Use setx to set user-level environment variable permanently
        let output = Command::new("setx")
            .args(["LLVM_SYS_140_PREFIX", &path_str])
            .output()?;

        if !output.status.success() {
            return Err(format!(
                "setx command failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        Ok(())
    }

    #[cfg(not(target_os = "windows"))]
    {
        // On Unix, we need to add to shell RC files for user or /etc/environment for system
        if system_wide {
            // System-wide: would need sudo to write to /etc/environment
            eprintln!(
                "Warning: System-wide environment variables on Unix require root privileges."
            );
            eprintln!("Setting user-level variable instead.");
            eprintln!();
            eprintln!("To set system-wide, add to /etc/environment (requires sudo):");
            eprintln!("  LLVM_SYS_140_PREFIX=\"{path_str}\"");
            eprintln!();
        }

        // User-level: append to shell RC files
        let home_dir = dirs::home_dir().ok_or("Could not determine home directory")?;
        let export_line = format!("export LLVM_SYS_140_PREFIX=\"{path_str}\"\n");

        // Try to add to common shell RC files
        let rc_files = vec![
            home_dir.join(".bashrc"),
            home_dir.join(".zshrc"),
            home_dir.join(".profile"),
        ];

        let mut updated_any = false;
        for rc_file in rc_files {
            if rc_file.exists() {
                // Check if already present
                if let Ok(contents) = fs::read_to_string(&rc_file)
                    && contents.contains("LLVM_SYS_140_PREFIX")
                {
                    println!(
                        "LLVM_SYS_140_PREFIX already present in {}",
                        rc_file.display()
                    );
                    updated_any = true;
                    continue;
                }

                // Append to file
                match fs::OpenOptions::new().append(true).open(&rc_file) {
                    Ok(mut file) => {
                        use std::io::Write;
                        writeln!(file, "\n# Added by pecos-llvm installer")?;
                        write!(file, "{export_line}")?;
                        println!("Added LLVM_SYS_140_PREFIX to {}", rc_file.display());
                        updated_any = true;
                    }
                    Err(e) => {
                        eprintln!("Warning: Could not update {}: {}", rc_file.display(), e);
                    }
                }
            }
        }

        if !updated_any {
            eprintln!("Warning: No shell RC files found. Please add manually:");
            eprintln!("  {}", export_line.trim());
            return Err("No shell RC files found to update".into());
        }

        Ok(())
    }
}

fn get_install_location(location: InstallLocation) -> Result<PathBuf, Box<dyn std::error::Error>> {
    match location {
        InstallLocation::PecosManaged => {
            // PECOS-managed: ~/.pecos/llvm (all platforms)
            let home_dir = dirs::home_dir().ok_or("Could not determine home directory")?;
            Ok(home_dir.join(".pecos").join("llvm"))
        }
        InstallLocation::System => {
            // System-wide: standard locations with admin/sudo rights
            #[cfg(target_os = "windows")]
            {
                Ok(PathBuf::from("C:\\Program Files\\LLVM-14"))
            }
            #[cfg(not(target_os = "windows"))]
            {
                Ok(PathBuf::from("/usr/local/LLVM-14"))
            }
        }
        InstallLocation::User => {
            // User-level: standard user-accessible locations
            #[cfg(target_os = "windows")]
            {
                // %LOCALAPPDATA%\Programs\LLVM-14
                let local_appdata = std::env::var("LOCALAPPDATA")
                    .map_err(|_| "Could not determine LOCALAPPDATA")?;
                Ok(PathBuf::from(local_appdata)
                    .join("Programs")
                    .join("LLVM-14"))
            }
            #[cfg(target_os = "macos")]
            {
                // ~/Library/Application Support/LLVM-14
                let home_dir = dirs::home_dir().ok_or("Could not determine home directory")?;
                Ok(home_dir
                    .join("Library")
                    .join("Application Support")
                    .join("LLVM-14"))
            }
            #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
            {
                // ~/.local/LLVM-14 (Linux and others)
                let home_dir = dirs::home_dir().ok_or("Could not determine home directory")?;
                Ok(home_dir.join(".local").join("LLVM-14"))
            }
        }
    }
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

// Allow unnecessary_wraps since the Result is needed on macOS but not on other platforms
#[allow(clippy::unnecessary_wraps)]
fn apply_platform_fixes(llvm_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "macos")]
    {
        print!("Applying macOS fixes... ");
        io::Write::flush(&mut io::stdout())?;

        // Fix 1: Configure clang to use macOS SDK
        // The pre-built LLVM doesn't know where system libraries are on macOS
        configure_macos_sdk(llvm_dir)?;

        // Fix 2: Fix libunwind dylib references
        // LLVM 14.0.6 libunwind libraries have @rpath references to themselves
        fix_libunwind_references(llvm_dir)?;

        println!("Done");
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Suppress unused parameter warning on non-macOS platforms
        let _ = llvm_dir;
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn configure_macos_sdk(llvm_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    // Get the SDK path from xcrun
    let output = Command::new("xcrun").args(["--show-sdk-path"]).output()?;

    if !output.status.success() {
        return Err("Failed to find macOS SDK. Is Xcode Command Line Tools installed?".into());
    }

    let sdk_path = String::from_utf8(output.stdout)?.trim().to_string();

    // Create wrapper scripts for clang and clang-14 that add SDK flags
    let bin_dir = llvm_dir.join("bin");

    // Rename original binaries
    let clang_orig = bin_dir.join("clang");
    let clang_real = bin_dir.join("clang-real");
    if clang_orig.exists() && !clang_real.exists() {
        fs::rename(&clang_orig, &clang_real)?;
    }

    let clang14_orig = bin_dir.join("clang-14");
    let clang14_real = bin_dir.join("clang-14-real");
    if clang14_orig.exists() && !clang14_real.exists() {
        fs::rename(&clang14_orig, &clang14_real)?;
    }

    // Create wrapper script for clang
    let wrapper_content = format!(
        "#!/bin/bash\nexec \"$(dirname \"$0\")/clang-real\" -isysroot {sdk_path} -L{sdk_path}/usr/lib \"$@\"\n"
    );
    fs::write(&clang_orig, &wrapper_content)?;
    fs::set_permissions(&clang_orig, fs::Permissions::from_mode(0o755))?;

    // Create wrapper script for clang-14
    let wrapper14_content = format!(
        "#!/bin/bash\nexec \"$(dirname \"$0\")/clang-14-real\" -isysroot {sdk_path} -L{sdk_path}/usr/lib \"$@\"\n"
    );
    fs::write(&clang14_orig, &wrapper14_content)?;
    fs::set_permissions(&clang14_orig, fs::Permissions::from_mode(0o755))?;

    // Also create wrapper for clang++ if it exists
    let clangpp_orig = bin_dir.join("clang++");
    let clangpp_real = bin_dir.join("clang++-real");
    if clangpp_orig.exists() && !clangpp_real.exists() {
        fs::rename(&clangpp_orig, &clangpp_real)?;
        let wrapperpp_content = format!(
            "#!/bin/bash\nexec \"$(dirname \"$0\")/clang++-real\" -isysroot {sdk_path} -L{sdk_path}/usr/lib \"$@\"\n"
        );
        fs::write(&clangpp_orig, &wrapperpp_content)?;
        fs::set_permissions(&clangpp_orig, fs::Permissions::from_mode(0o755))?;
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn fix_libunwind_references(llvm_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;

    let lib_dir = llvm_dir.join("lib");

    // Find all libunwind dylibs
    if let Ok(entries) = fs::read_dir(&lib_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                // Check for libunwind dylib files using case-insensitive extension check
                if name.starts_with("libunwind")
                    && std::path::Path::new(name)
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("dylib"))
                {
                    // Fix the install name to use absolute path instead of @rpath
                    let abs_path = path.to_string_lossy().to_string();
                    Command::new("install_name_tool")
                        .args(["-id", &abs_path, &abs_path])
                        .output()?;
                }
            }
        }
    }

    Ok(())
}
