//! LLVM 14.0.6 installation functionality
//!
//! Downloads and extracts LLVM 14.0.6 pre-built binaries to a project-local directory.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Install LLVM 14.0.6 to user data directory
///
/// Installs to platform-appropriate location:
/// - macOS: ~/Library/Application Support/pecos/llvm
/// - Linux: ~/.local/share/pecos/llvm
/// - Windows: %LOCALAPPDATA%/pecos/llvm
///
/// # Arguments
/// * `force` - Force reinstall even if already present
/// * `no_configure` - Skip automatic configuration after installation
///
/// # Errors
/// Returns an error if:
/// - LLVM is already installed and `force` is false
/// - The download or extraction fails
/// - Platform-specific fixes fail
/// - Installation verification fails
///
/// # Returns
/// Path to the installed LLVM directory
pub fn install_llvm(
    force: bool,
    no_configure: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let llvm_dir = get_install_location()?;

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

    // Extract
    extract_llvm(&archive_path, &llvm_dir)?;

    // Cleanup LLVM temp directory only (not entire tmp directory)
    fs::remove_dir_all(&temp_dir)?;

    // Apply platform-specific fixes
    apply_platform_fixes(&llvm_dir)?;

    // Verify
    if !is_valid_installation(&llvm_dir) {
        return Err("Installation completed but verification failed".into());
    }

    println!();
    println!("Installation complete!");
    println!("LLVM 14.0.6 installed to: {}", llvm_dir.display());

    // Auto-configure LLVM for PECOS (unless --no-configure is specified)
    if no_configure {
        println!();
        println!("Skipping automatic configuration (--no-configure specified).");
        println!();
        println!("To configure PECOS, run:");
        println!("  cargo run -p pecos-llvm-utils --bin pecos-llvm -- configure");
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
                println!("  cargo run -p pecos-llvm-utils --bin pecos-llvm -- configure");
            }
        }
    }

    Ok(llvm_dir)
}

fn get_install_location() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let home_dir = dirs::home_dir().ok_or("Could not determine home directory")?;

    Ok(home_dir.join(".pecos").join("llvm"))
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

    // The archive extracts to a directory like LLVM-14.0.6-win64
    // Find the extracted directory
    let mut extracted_dir = None;
    for entry in fs::read_dir(extract_to)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && path.file_name().unwrap().to_str().unwrap().contains("LLVM") {
            extracted_dir = Some(path);
            break;
        }
    }

    let extracted_dir = extracted_dir.ok_or("Could not find extracted LLVM directory")?;

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

fn is_valid_installation(path: &Path) -> bool {
    let llvm_config = if cfg!(windows) {
        path.join("bin").join("llvm-config.exe")
    } else {
        path.join("bin").join("llvm-config")
    };

    llvm_config.exists()
}

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
