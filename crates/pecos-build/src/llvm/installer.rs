//! LLVM 21.1 installation functionality

#![allow(clippy::case_sensitive_file_extension_comparisons)]

use crate::errors::{Error, Result};
#[cfg(target_os = "linux")]
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const LLVM_RELEASE_VERSION: &str = "21.1.8";

/// LLVM release version installed by the managed installer.
#[must_use]
pub const fn release_version() -> &'static str {
    LLVM_RELEASE_VERSION
}

/// Explain why PECOS cannot provide a managed shared LLVM install here.
#[must_use]
pub fn managed_install_unavailable_reason() -> Option<&'static str> {
    #[cfg(target_os = "linux")]
    {
        None
    }

    #[cfg(target_os = "macos")]
    {
        Some(
            "PECOS-managed shared LLVM is currently available only on \
             Debian/Ubuntu-compatible Linux. On macOS, install LLVM 21 with \
             Homebrew (`brew install llvm@21`) and run `pecos llvm configure`.",
        )
    }

    #[cfg(target_os = "windows")]
    {
        Some(
            "PECOS-managed LLVM is not implemented in the CLI on Windows yet. \
             Use `scripts\\ci\\install-llvm-21-windows.ps1` to install the \
             conda-forge LLVM 21.1 toolchain under `%USERPROFILE%\\.pecos\\deps`, \
             then run `pecos llvm configure \
             %USERPROFILE%\\.pecos\\deps\\llvm-21.1\\Library`, or configure \
             your own full LLVM 21.1 install.",
        )
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Some(
            "PECOS-managed shared LLVM is currently available only on \
             Debian/Ubuntu-compatible Linux. Install shared LLVM 21 with your \
             system package manager and run `pecos llvm configure /path/to/llvm`.",
        )
    }
}

#[cfg(target_os = "linux")]
const APT_LLVM_PACKAGES: &[&str] = &[
    "libllvm21",
    "llvm-21",
    "llvm-21-dev",
    "llvm-21-linker-tools",
    "clang-21",
    "libclang-common-21-dev",
    "libclang-cpp21",
    "libclang1-21",
];

#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
struct AptLlvmSource {
    base_url: String,
    codename: String,
    deb_arch: String,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
struct AptPackage {
    name: String,
    version: String,
    filename: String,
    sha256: String,
}

/// Install LLVM 21.1 to `~/.pecos/deps/llvm-21.1/`
///
/// # Arguments
/// * `force` - Force reinstall even if already present
/// * `no_configure` - Skip automatic configuration after installation
///
/// # Errors
///
/// Returns an error if installation fails
pub fn install_llvm(force: bool, no_configure: bool) -> Result<PathBuf> {
    // Always install to the versioned path
    let llvm_dir = crate::home::get_versioned_dep_path("llvm", crate::home::LLVM_VERSION)?;

    // Check if already installed
    if llvm_dir.exists() {
        if is_valid_installation(&llvm_dir) {
            if is_shared_installation(&llvm_dir) {
                if !force {
                    return Err(Error::Llvm(
                        "LLVM is already installed. Use --force to reinstall.".into(),
                    ));
                }
            } else if !force {
                let recovery = managed_install_unavailable_reason().map_or_else(
                    || {
                        "Run `pecos install llvm --force` to replace it with shared LLVM, \
                         or configure your own shared LLVM with `pecos llvm configure /path/to/llvm`."
                            .to_string()
                    },
                    |reason| {
                        format!(
                            "{reason} Remove the static managed install manually if you no longer need it."
                        )
                    },
                );
                return Err(Error::Llvm(format!(
                    "Existing PECOS-managed LLVM is static. {recovery}"
                )));
            }
        } else if !force {
            let recovery = managed_install_unavailable_reason().map_or_else(
                || "Use --force to reinstall.".to_string(),
                |reason| {
                    format!("{reason} Remove the invalid managed install manually if you no longer need it.")
                },
            );
            return Err(Error::Llvm(format!(
                "Existing LLVM directory is not a valid LLVM {} installation: {}. {recovery}",
                super::REQUIRED_VERSION,
                llvm_dir.display()
            )));
        }
    }

    if let Some(reason) = managed_install_unavailable_reason() {
        return Err(Error::Llvm(reason.into()));
    }

    // Remove existing if force
    if force && llvm_dir.exists() {
        println!("Removing existing LLVM installation...");
        fs::remove_dir_all(&llvm_dir)?;
    }

    println!("Installing LLVM {LLVM_RELEASE_VERSION}...");
    println!("PECOS-managed LLVM uses a shared LLVM library when available.");
    println!("This downloads a large toolchain and may take several minutes.");
    println!();

    // Create parent directory
    if let Some(parent) = llvm_dir.parent() {
        fs::create_dir_all(parent)?;
    }

    // Download to temp directory
    let temp_base = llvm_dir.parent().unwrap_or(&llvm_dir).join("tmp");
    let temp_dir = temp_base.join("llvm");
    fs::create_dir_all(&temp_dir)?;

    install_managed_llvm_payload(&llvm_dir, &temp_dir)?;

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
    if !is_shared_installation(&llvm_dir) {
        return Err(Error::Llvm(
            "Managed LLVM installation is static. PECOS requires managed LLVM to provide \
             shared libLLVM; configure a shared system LLVM with `pecos llvm configure /path/to/llvm`."
                .into(),
        ));
    }

    verify_llvm_runtime(&llvm_dir)?;

    println!();
    println!("Installation complete!");
    println!(
        "LLVM {LLVM_RELEASE_VERSION} installed to: {}",
        llvm_dir.display()
    );

    if no_configure {
        println!();
        println!("Skipping automatic configuration (--no-configure specified).");
        println!();
        println!("To configure PECOS, run:");
        println!("  pecos llvm configure");
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
                println!("  pecos llvm configure");
            }
        }
    }

    Ok(llvm_dir)
}

fn install_managed_llvm_payload(llvm_dir: &Path, temp_dir: &Path) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        if let Some(source) = detect_apt_llvm_source() {
            return install_linux_apt_llvm(llvm_dir, temp_dir, &source);
        }

        Err(Error::Llvm(
            "PECOS-managed shared LLVM on Linux currently requires a Debian/Ubuntu-compatible \
             apt.llvm.org repository. Install shared LLVM 21 with your system package manager and \
             run `pecos llvm configure /path/to/llvm`."
                .into(),
        ))
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (llvm_dir, temp_dir);
        Err(Error::Llvm(
            managed_install_unavailable_reason()
                .unwrap_or("PECOS-managed shared LLVM is not available on this platform.")
                .into(),
        ))
    }
}

#[cfg(target_os = "linux")]
fn detect_apt_llvm_source() -> Option<AptLlvmSource> {
    let os_release = fs::read_to_string("/etc/os-release").ok()?;
    let codename = os_release_value(&os_release, "UBUNTU_CODENAME")
        .or_else(|| os_release_value(&os_release, "VERSION_CODENAME"))?;
    let deb_arch = match std::env::consts::ARCH {
        "x86_64" => "amd64",
        "aarch64" => "arm64",
        _ => return None,
    };

    Some(AptLlvmSource {
        base_url: "https://apt.llvm.org".to_string(),
        codename,
        deb_arch: deb_arch.to_string(),
    })
}

#[cfg(target_os = "linux")]
fn os_release_value(contents: &str, key: &str) -> Option<String> {
    contents.lines().find_map(|line| {
        let (name, value) = line.split_once('=')?;
        if name != key {
            return None;
        }
        let value = value.trim().trim_matches('"').to_string();
        (!value.is_empty()).then_some(value)
    })
}

#[cfg(target_os = "linux")]
fn install_linux_apt_llvm(llvm_dir: &Path, temp_dir: &Path, source: &AptLlvmSource) -> Result<()> {
    println!(
        "Using apt.llvm.org shared LLVM packages for {} {}.",
        source.codename, source.deb_arch
    );

    let packages_url = format!(
        "{}/{}/dists/llvm-toolchain-{}-21/main/binary-{}/Packages.gz",
        source.base_url, source.codename, source.codename, source.deb_arch
    );
    let packages_gz = temp_dir.join("Packages.gz");
    download_file(&packages_url, &packages_gz, "LLVM package index")?;

    let packages = read_apt_packages(&packages_gz)?;
    let selected = select_apt_packages(&packages)?;
    let deb_dir = temp_dir.join("debs");
    let root_dir = temp_dir.join("root");
    fs::create_dir_all(&deb_dir)?;
    fs::create_dir_all(&root_dir)?;

    for package in selected {
        let url = format!(
            "{}/{}/{}",
            source.base_url, source.codename, package.filename
        );
        let filename = Path::new(&package.filename).file_name().ok_or_else(|| {
            Error::Archive(format!("Invalid package filename: {}", package.filename))
        })?;
        let deb_path = deb_dir.join(filename);
        println!("Downloading {} {}...", package.name, package.version);
        download_file(&url, &deb_path, &package.name)?;
        verify_checksum_value(&deb_path, &package.sha256)?;
        extract_deb_data(&deb_path, &root_dir)?;
    }

    install_apt_root_as_llvm_prefix(&root_dir, llvm_dir, &source.deb_arch)?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn read_apt_packages(packages_gz: &Path) -> Result<Vec<AptPackage>> {
    let file = fs::File::open(packages_gz)?;
    let mut decoder = flate2::read::GzDecoder::new(file);
    let mut contents = String::new();
    io::Read::read_to_string(&mut decoder, &mut contents)?;

    let mut packages = Vec::new();
    for stanza in contents.split("\n\n") {
        let mut name = None;
        let mut version = None;
        let mut filename = None;
        let mut sha256 = None;

        for line in stanza.lines() {
            if let Some(value) = line.strip_prefix("Package: ") {
                name = Some(value.trim().to_string());
            } else if let Some(value) = line.strip_prefix("Version: ") {
                version = Some(value.trim().to_string());
            } else if let Some(value) = line.strip_prefix("Filename: ") {
                filename = Some(value.trim().to_string());
            } else if let Some(value) = line.strip_prefix("SHA256: ") {
                sha256 = Some(value.trim().to_string());
            }
        }

        if let (Some(name), Some(version), Some(filename), Some(sha256)) =
            (name, version, filename, sha256)
        {
            packages.push(AptPackage {
                name,
                version,
                filename,
                sha256,
            });
        }
    }

    Ok(packages)
}

#[cfg(target_os = "linux")]
fn select_apt_packages(packages: &[AptPackage]) -> Result<Vec<AptPackage>> {
    let mut selected = Vec::with_capacity(APT_LLVM_PACKAGES.len());
    for package_name in APT_LLVM_PACKAGES {
        let package = packages
            .iter()
            .rev()
            .find(|package| {
                package.name == *package_name && package.version.contains(LLVM_RELEASE_VERSION)
            })
            .ok_or_else(|| {
                Error::Llvm(format!(
                    "apt.llvm.org package {package_name} {LLVM_RELEASE_VERSION} was not found"
                ))
            })?;
        selected.push(package.clone());
    }
    Ok(selected)
}

#[cfg(target_os = "linux")]
fn download_file(url: &str, dest: &Path, label: &str) -> Result<()> {
    print!("Downloading {label}... ");
    io::Write::flush(&mut io::stdout())?;

    let response = reqwest::blocking::get(url).map_err(|e| Error::Http(e.to_string()))?;
    if !response.status().is_success() {
        return Err(Error::Http(format!(
            "GET {url} failed: {}",
            response.status()
        )));
    }
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
                print!("\rDownloading {label}... {progress:.0}%");
                io::Write::flush(&mut io::stdout())?;
                last_print = progress;
            }
        }
    }

    println!(
        "\rDownloading {label}... Done ({} MB)",
        downloaded / 1_000_000
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn verify_checksum_value(file_path: &Path, expected: &str) -> Result<()> {
    print!("Verifying checksum... ");
    io::Write::flush(&mut io::stdout())?;

    let computed_hash = sha256_file(file_path)?;
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

#[cfg(target_os = "linux")]
fn sha256_file(file_path: &Path) -> Result<String> {
    let data = fs::read(file_path)?;
    let mut hasher = Sha256::new();
    Digest::update(&mut hasher, &data);
    Ok(hasher.finalize().iter().fold(String::new(), |mut s, b| {
        use std::fmt::Write;
        write!(s, "{b:02x}").unwrap();
        s
    }))
}

#[cfg(target_os = "linux")]
fn extract_deb_data(deb_path: &Path, dest: &Path) -> Result<()> {
    let data = fs::read(deb_path)?;
    if !data.starts_with(b"!<arch>\n") {
        return Err(Error::Archive(format!(
            "{} is not a Debian ar archive",
            deb_path.display()
        )));
    }

    let mut offset = 8usize;
    while offset + 60 <= data.len() {
        let header = &data[offset..offset + 60];
        offset += 60;

        let name = String::from_utf8_lossy(&header[0..16])
            .trim()
            .trim_end_matches('/')
            .to_string();
        let size_text = String::from_utf8_lossy(&header[48..58]).trim().to_string();
        let size: usize = size_text.parse().map_err(|e| {
            Error::Archive(format!(
                "Invalid ar member size in {}: {e}",
                deb_path.display()
            ))
        })?;
        if offset + size > data.len() {
            return Err(Error::Archive(format!(
                "Truncated ar member in {}",
                deb_path.display()
            )));
        }

        let member = &data[offset..offset + size];
        offset += size + (size % 2);

        match name.as_str() {
            "data.tar.zst" => {
                let decoder = zstd::stream::read::Decoder::new(member)?;
                let mut archive = tar::Archive::new(decoder);
                archive.unpack(dest)?;
                return Ok(());
            }
            "data.tar.xz" => {
                let decoder = xz2::read::XzDecoder::new(member);
                let mut archive = tar::Archive::new(decoder);
                archive.unpack(dest)?;
                return Ok(());
            }
            "data.tar.gz" => {
                let decoder = flate2::read::GzDecoder::new(member);
                let mut archive = tar::Archive::new(decoder);
                archive.unpack(dest)?;
                return Ok(());
            }
            _ => {}
        }
    }

    Err(Error::Archive(format!(
        "{} did not contain a supported data.tar member",
        deb_path.display()
    )))
}

#[cfg(target_os = "linux")]
fn install_apt_root_as_llvm_prefix(root_dir: &Path, llvm_dir: &Path, deb_arch: &str) -> Result<()> {
    let apt_prefix = root_dir.join("usr").join("lib").join("llvm-21");
    if !apt_prefix.exists() {
        return Err(Error::Archive(format!(
            "apt.llvm.org packages did not provide {}",
            apt_prefix.display()
        )));
    }

    if llvm_dir.exists() {
        fs::remove_dir_all(llvm_dir)?;
    }
    fs::rename(&apt_prefix, llvm_dir)?;

    let gnu_arch = match deb_arch {
        "amd64" => "x86_64-linux-gnu",
        "arm64" => "aarch64-linux-gnu",
        _ => {
            return Err(Error::Archive(format!(
                "Unsupported Debian architecture: {deb_arch}"
            )));
        }
    };
    let system_lib_dir = root_dir.join("usr").join("lib").join(gnu_arch);
    let llvm_lib_dir = llvm_dir.join("lib");
    fs::create_dir_all(&llvm_lib_dir)?;

    install_apt_headers(root_dir, llvm_dir)?;
    copy_shared_libraries(&system_lib_dir, &llvm_lib_dir)?;
    fix_local_shared_library_links(&llvm_lib_dir)?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn install_apt_headers(root_dir: &Path, llvm_dir: &Path) -> Result<()> {
    let include_dir = llvm_dir.join("include");
    fs::create_dir_all(&include_dir)?;

    copy_dir_contents(
        &root_dir
            .join("usr")
            .join("include")
            .join("llvm-21")
            .join("llvm"),
        &include_dir.join("llvm"),
    )?;
    copy_dir_contents(
        &root_dir
            .join("usr")
            .join("include")
            .join("llvm-c-21")
            .join("llvm-c"),
        &include_dir.join("llvm-c"),
    )?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn copy_dir_contents(source: &Path, dest: &Path) -> Result<()> {
    if !source.exists() {
        return Err(Error::Archive(format!(
            "apt.llvm.org packages did not provide {}",
            source.display()
        )));
    }

    if dest.exists() || fs::symlink_metadata(dest).is_ok() {
        let metadata = fs::symlink_metadata(dest)?;
        if metadata.is_dir() {
            fs::remove_dir_all(dest)?;
        } else {
            fs::remove_file(dest)?;
        }
    }
    fs::create_dir_all(dest)?;

    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            copy_dir_contents(&source_path, &dest_path)?;
        } else if file_type.is_file() {
            fs::copy(&source_path, &dest_path)?;
        } else if file_type.is_symlink() {
            let target = fs::read_link(&source_path)?;
            std::os::unix::fs::symlink(target, dest_path)?;
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn copy_shared_libraries(source_dir: &Path, dest_dir: &Path) -> Result<()> {
    if !source_dir.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !file_name.starts_with("lib") || !file_name.contains(".so") {
            continue;
        }

        let dest = dest_dir.join(file_name);
        if dest.exists() || fs::symlink_metadata(&dest).is_ok() {
            let _ = fs::remove_file(&dest);
        }
        fs::copy(&path, dest)?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn fix_local_shared_library_links(lib_dir: &Path) -> Result<()> {
    replace_symlink(lib_dir, "libLLVM-21.so", "libLLVM.so.21.1")?;
    replace_symlink(lib_dir, "libLLVM.so", "libLLVM.so.21.1")?;
    replace_symlink(lib_dir, "libLLVM.so.1", "libLLVM.so.21.1")?;

    if lib_dir.join("libclang-cpp.so.21.1").exists() {
        replace_symlink(lib_dir, "libclang-cpp.so.21", "libclang-cpp.so.21.1")?;
        replace_symlink(lib_dir, "libclang-cpp.so", "libclang-cpp.so.21.1")?;
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn replace_symlink(lib_dir: &Path, link: &str, target: &str) -> Result<()> {
    use std::os::unix::fs::symlink;

    let link_path = lib_dir.join(link);
    if link_path.exists() || fs::symlink_metadata(&link_path).is_ok() {
        let _ = fs::remove_file(&link_path);
    }
    symlink(target, link_path)?;
    Ok(())
}

/// Remove local LLVM installation (`~/.pecos/deps/llvm-{version}/`)
///
/// # Errors
///
/// Returns an error if removal fails
pub fn uninstall_llvm() -> Result<()> {
    let llvm_dir = crate::home::get_llvm_dir_path()?;

    if !llvm_dir.exists() {
        println!(
            "LLVM is not installed in ~/.pecos/deps/llvm-{}/",
            super::REQUIRED_VERSION
        );
        return Ok(());
    }

    println!("Removing LLVM installation at: {}", llvm_dir.display());
    fs::remove_dir_all(&llvm_dir)?;
    println!("LLVM uninstalled successfully");

    Ok(())
}

/// Validate that a path contains a complete LLVM installation
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

    super::get_llvm_version(path).is_ok_and(|version| super::is_required_llvm_version(&version))
}

/// Return whether an LLVM installation reports shared libLLVM support.
#[must_use]
pub fn is_shared_installation(path: &Path) -> bool {
    super::get_llvm_shared_mode(path).is_ok_and(|mode| mode.trim().eq_ignore_ascii_case("shared"))
        && super::get_llvm_shared_libraries(path).is_some()
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
        if super::is_required_llvm_version(&version) {
            let link_mode =
                super::get_llvm_shared_mode(llvm_dir).unwrap_or_else(|_| "unknown".into());
            println!("OK (version {version}, link mode {link_mode})");
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

#[cfg(target_os = "linux")]
fn apply_platform_fixes(llvm_dir: &Path) -> Result<()> {
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::process::Command;

    print!("Applying Linux platform fixes... ");
    io::Write::flush(&mut io::stdout())?;

    let llvm_config = llvm_dir.join("bin").join("llvm-config");
    let llvm_config_real = llvm_dir.join("bin").join("llvm-config.real");
    let gnu_arch = match std::env::consts::ARCH {
        "x86_64" => "x86_64-linux-gnu",
        "aarch64" => "aarch64-linux-gnu",
        _ => {
            println!("Skipped (unsupported architecture)");
            return Ok(());
        }
    };
    let missing_zstd_static = format!("/usr/lib/{gnu_arch}/libzstd.a");
    let system_zstd_runtime = [
        PathBuf::from(format!("/lib/{gnu_arch}/libzstd.so.1")),
        PathBuf::from(format!("/usr/lib/{gnu_arch}/libzstd.so.1")),
    ]
    .into_iter()
    .find(|path| path.exists());

    let Ok(output) = Command::new(&llvm_config)
        .args(["--system-libs", "--link-static"])
        .output()
    else {
        println!("Skipped (llvm-config unavailable)");
        return Ok(());
    };

    let system_libs = String::from_utf8_lossy(&output.stdout);
    let Some(system_zstd_runtime) = system_zstd_runtime else {
        println!("Skipped");
        return Ok(());
    };

    if !system_libs.contains(&missing_zstd_static) || Path::new(&missing_zstd_static).exists() {
        println!("Skipped");
        return Ok(());
    }

    let local_zstd = llvm_dir.join("lib").join("libzstd.so");
    if !local_zstd.exists() {
        symlink(&system_zstd_runtime, &local_zstd)?;
    }

    if !llvm_config_real.exists() {
        fs::rename(&llvm_config, &llvm_config_real)?;
    }

    let escaped_zstd_static = missing_zstd_static.replace('/', "\\/");
    let wrapper = r#"#!/usr/bin/env bash
real="$(dirname "$0")/llvm-config.real"
output="$("$real" "$@")"
status=$?
if [ "$status" -ne 0 ]; then
    exit "$status"
fi
case " $* " in
    *" --system-libs "*)
        output="${output//__PECOS_ZSTD_STATIC__/-lzstd}"
        ;;
esac
printf '%s\n' "$output"
"#
    .replace("__PECOS_ZSTD_STATIC__", &escaped_zstd_static);
    fs::write(&llvm_config, wrapper)?;
    let mut permissions = fs::metadata(&llvm_config)?.permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&llvm_config, permissions)?;

    println!("OK");
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
#[allow(clippy::unnecessary_wraps)]
fn apply_platform_fixes(_llvm_dir: &Path) -> Result<()> {
    Ok(())
}
