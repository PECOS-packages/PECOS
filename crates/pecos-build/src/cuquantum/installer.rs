//! cuQuantum SDK installation functionality
//!
//! Downloads and installs cuQuantum SDK to `~/.pecos/deps/cuquantum/`

use crate::errors::{Error, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use super::{CUQUANTUM_VERSION, config, get_pecos_cuquantum_dir, is_valid_cuquantum_installation};

/// cuQuantum download information
struct CuQuantumDownload {
    url: String,
    filename: String,
    /// Expected SHA256 of the archive. Required: an unverified multi-hundred-megabyte
    /// archive is extracted into `~/.pecos`, so there is no safe "skip" path.
    sha256: &'static str,
}

/// SHA256 checksums for the [`CUQUANTUM_VERSION`] archives, by platform key and CUDA major.
///
/// NVIDIA publishes these in the release manifest alongside the redistributable archives.
/// To refresh when bumping [`CUQUANTUM_VERSION`], read the `sha256` fields from
/// `https://developer.download.nvidia.com/compute/cuquantum/redist/redistrib_<release>.json`
/// (the release label omits the trailing build component, so `25.11.1.11` is published as
/// `redistrib_25.11.1.json`). Values must come from that manifest, never from a local
/// download, so that a substituted archive cannot certify itself.
const CUQUANTUM_CHECKSUMS: &[(&str, u32, &str)] = &[
    (
        "linux-x86_64",
        12,
        "77138174ece7a9a9108e2d44b8ea1e1f2e29874544ea940e13c0d607ad15a358",
    ),
    (
        "linux-x86_64",
        13,
        "dfc063a88547636b316b9682a6834c5fbd2dfd7e823c5fc58857c4251d2ab3c2",
    ),
    (
        "linux-sbsa",
        12,
        "3d3cb33268d97cbe48eea426ea44238ed55c622279fd694f0cf32e47c4564454",
    ),
    (
        "linux-sbsa",
        13,
        "14a44b09d384c27dd36eae9d9858c90140e6422bbd80bda1041ded96ec3c7223",
    ),
];

/// Looks up the published checksum for a platform and CUDA major version.
fn checksum_for(platform: &str, cuda_major: u32) -> Result<&'static str> {
    CUQUANTUM_CHECKSUMS
        .iter()
        .find(|(key, major, _)| *key == platform && *major == cuda_major)
        .map(|(_, _, hash)| *hash)
        .ok_or_else(|| {
            Error::CuQuantum(format!(
                "no published cuQuantum {CUQUANTUM_VERSION} archive for {platform} with CUDA {cuda_major}; \
                 supported combinations are {}",
                CUQUANTUM_CHECKSUMS
                    .iter()
                    .map(|(key, major, _)| format!("{key}/CUDA {major}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
}

/// Detect CUDA major version from installed CUDA
fn detect_cuda_version() -> u32 {
    if let Some(cuda_path) = crate::cuda::find_cuda()
        && let Ok(version) = crate::cuda::get_cuda_version(&cuda_path)
    {
        // Parse version like "12.4" or "11.8"
        if let Some(major) = version.split('.').next()
            && let Ok(v) = major.parse::<u32>()
        {
            return v;
        }
    }
    // Default to CUDA 12 if not detected (most modern systems)
    12
}

/// Get download URL for the current platform
fn get_download_info() -> Result<CuQuantumDownload> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let cuda_major = detect_cuda_version();

    // cuQuantum download URLs follow pattern:
    // https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-25.03.0.11_cuda12-archive.tar.xz
    //
    // Each supported platform must have a published checksum in CUQUANTUM_CHECKSUMS;
    // checksum_for rejects any combination NVIDIA does not publish.

    match (os, arch) {
        ("linux", "x86_64") => Ok(CuQuantumDownload {
            url: format!(
                "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-{CUQUANTUM_VERSION}_cuda{cuda_major}-archive.tar.xz"
            ),
            filename: format!(
                "cuquantum-linux-x86_64-{CUQUANTUM_VERSION}_cuda{cuda_major}-archive.tar.xz"
            ),
            sha256: checksum_for("linux-x86_64", cuda_major)?,
        }),
        ("linux", "aarch64") => Ok(CuQuantumDownload {
            url: format!(
                "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-sbsa/cuquantum-linux-sbsa-{CUQUANTUM_VERSION}_cuda{cuda_major}-archive.tar.xz"
            ),
            filename: format!(
                "cuquantum-linux-sbsa-{CUQUANTUM_VERSION}_cuda{cuda_major}-archive.tar.xz"
            ),
            sha256: checksum_for("linux-sbsa", cuda_major)?,
        }),
        // NVIDIA publishes no windows-x86_64 archive for this release: the whole
        // redist/cuquantum/windows-x86_64/ path returns 404, so the previous download URL
        // could never succeed. Fail with the reason instead of a bare download error.
        ("windows", "x86_64") => Err(Error::CuQuantum(format!(
            "NVIDIA does not publish a Windows cuQuantum archive for {CUQUANTUM_VERSION}; \
             install cuQuantum through the NVIDIA-provided channels for Windows"
        ))),
        ("macos", _) => Err(Error::CuQuantum(
            "cuQuantum is not supported on macOS (NVIDIA GPUs not supported)".into(),
        )),
        _ => Err(Error::CuQuantum(format!(
            "Unsupported platform: {os}/{arch}"
        ))),
    }
}

/// Install cuQuantum SDK to `~/.pecos/cuquantum/`
///
/// # Arguments
/// * `force` - Force reinstall even if already present
///
/// # Errors
/// Returns an error if:
/// - Home directory cannot be determined
/// - cuQuantum is already installed (unless `force` is true)
/// - Platform is unsupported
/// - Download or extraction fails
/// - Installation verification fails
pub fn install_cuquantum(force: bool) -> Result<PathBuf> {
    let cuquantum_dir = crate::home::get_versioned_dep_path("cuquantum", super::CUQUANTUM_VERSION)?;

    // Check if already installed
    if !force && cuquantum_dir.exists() && is_valid_cuquantum_installation(&cuquantum_dir) {
        return Err(Error::CuQuantum(
            "cuQuantum is already installed. Use --force to reinstall.".into(),
        ));
    }

    // Clean up invalid existing installation before re-downloading
    if !force && cuquantum_dir.exists() && !is_valid_cuquantum_installation(&cuquantum_dir) {
        fs::remove_dir_all(&cuquantum_dir)?;
    }

    // Check CUDA availability
    let cuda_version = if crate::cuda::is_cuda_available() {
        detect_cuda_version()
    } else {
        println!("Warning: CUDA not found. cuQuantum requires CUDA to function.");
        println!("Consider installing CUDA first with: pecos install cuda");
        println!();
        12 // Default to CUDA 12
    };

    // Remove existing if force
    if force && cuquantum_dir.exists() {
        println!("Removing existing cuQuantum installation...");
        fs::remove_dir_all(&cuquantum_dir)?;
    }

    let download_info = get_download_info()?;

    println!("Installing cuQuantum SDK {CUQUANTUM_VERSION} for CUDA {cuda_version}...");
    println!("This will download ~50-70MB and may take a few minutes.");
    println!();
    println!("Note: By downloading cuQuantum, you agree to NVIDIA's license terms.");
    println!("See: https://docs.nvidia.com/cuda/cuquantum/latest/license.html");
    println!();

    // Create cache directory
    let cache_dir = cuquantum_dir
        .parent()
        .ok_or_else(|| Error::CuQuantum("Invalid cuQuantum directory".into()))?
        .join("cache");
    fs::create_dir_all(&cache_dir)?;

    let archive_path = cache_dir.join(&download_info.filename);

    // Download if not already cached
    if archive_path.exists() {
        println!("Using cached download: {}", archive_path.display());
    } else {
        download_cuquantum(&download_info.url, &archive_path)?;
    }

    // Verify on every install, not only after a fresh download: the cache lives on disk
    // between runs, so a truncated or tampered cached archive must not be trusted just
    // because some earlier run put it there.
    verify_checksum(&archive_path, download_info.sha256)?;

    // Extract cuQuantum
    extract_cuquantum(&archive_path, &cuquantum_dir)?;

    // Verify installation
    if !is_valid_cuquantum_installation(&cuquantum_dir) {
        return Err(Error::CuQuantum(
            "Installation completed but verification failed".into(),
        ));
    }

    // Write version marker
    let version_file = cuquantum_dir.join("version.txt");
    fs::write(
        &version_file,
        format!("cuQuantum {CUQUANTUM_VERSION}\nInstalled by pecos\n"),
    )?;

    println!();
    println!("Installation complete!");
    println!(
        "cuQuantum SDK {} installed to: {}",
        CUQUANTUM_VERSION,
        cuquantum_dir.display()
    );

    // Try to auto-configure .cargo/config.toml
    match config::auto_configure_cuquantum(None) {
        Ok(path) => {
            println!();
            println!(
                "Configured .cargo/config.toml with CUQUANTUM_ROOT={}",
                path.display()
            );
        }
        Err(e) => {
            println!();
            println!("Note: Could not auto-configure .cargo/config.toml: {e}");
            println!("You may need to set the environment variable manually:");
            println!("  export CUQUANTUM_ROOT=\"{}\"", cuquantum_dir.display());
        }
    }

    Ok(cuquantum_dir)
}

/// Download cuQuantum archive
fn download_cuquantum(url: &str, dest: &Path) -> Result<()> {
    print!("Downloading cuQuantum SDK... ");
    io::stdout().flush()?;

    crate::download::ensure_crypto_provider();
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
        let mut buffer = vec![0; 65536]; // 64KB buffer
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
                print!("\rDownloading cuQuantum SDK... {progress:.0}%");
                io::stdout().flush()?;
                last_print = progress;
            }
        }
    }

    println!(
        "\rDownloading cuQuantum SDK... Done ({} MB)",
        downloaded / 1_000_000
    );
    Ok(())
}

/// Verify file checksum
fn verify_checksum(file_path: &Path, expected: &str) -> Result<()> {
    print!("Verifying checksum... ");
    io::stdout().flush()?;

    let data = fs::read(file_path)?;
    let mut hasher = Sha256::new();
    Digest::update(&mut hasher, &data);
    let computed_hash = hasher.finalize().iter().fold(String::new(), |mut s, b| {
        use std::fmt::Write;
        write!(s, "{b:02x}").unwrap();
        s
    });

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

/// Extract cuQuantum from archive
fn extract_cuquantum(archive: &Path, dest: &Path) -> Result<()> {
    let filename = archive
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Error::Archive("Invalid archive path".into()))?;

    println!("Extracting cuQuantum SDK...");

    fs::create_dir_all(dest)?;

    if filename.ends_with(".tar.xz") {
        extract_tar_xz(archive, dest)
    } else if std::path::Path::new(filename)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("zip"))
    {
        extract_zip(archive, dest)
    } else {
        Err(Error::Archive(format!(
            "Unsupported archive format: {filename}"
        )))
    }
}

/// Extract .tar.xz archive
fn extract_tar_xz(archive: &Path, dest: &Path) -> Result<()> {
    use std::process::Command;

    // Use system tar for .tar.xz (most reliable)
    let status = Command::new("tar")
        .arg("-xf")
        .arg(archive)
        .arg("-C")
        .arg(dest)
        .arg("--strip-components=1") // Remove top-level directory
        .status()
        .map_err(|e| Error::Archive(format!("Failed to run tar: {e}")))?;

    if !status.success() {
        return Err(Error::Archive("tar extraction failed".into()));
    }

    println!("Done");
    Ok(())
}

/// Extract .zip archive using system unzip
fn extract_zip(archive: &Path, dest: &Path) -> Result<()> {
    use std::process::Command;

    // Create a temp directory for extraction
    let temp_dir = dest
        .parent()
        .ok_or_else(|| Error::Archive("Invalid destination path".into()))?
        .join("tmp")
        .join("cuquantum_extract");

    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)?;
    }
    fs::create_dir_all(&temp_dir)?;

    // Use system unzip
    let status = Command::new("unzip")
        .arg("-q") // quiet
        .arg(archive)
        .arg("-d")
        .arg(&temp_dir)
        .status()
        .map_err(|e| Error::Archive(format!("Failed to run unzip: {e}")))?;

    if !status.success() {
        return Err(Error::Archive("unzip extraction failed".into()));
    }

    // Find the extracted directory (should be one top-level dir)
    let entries: Vec<_> = fs::read_dir(&temp_dir)?.flatten().collect();
    if entries.len() == 1 && entries[0].path().is_dir() {
        // Move contents from subdirectory to dest
        let subdir = entries[0].path();
        for entry in fs::read_dir(&subdir)?.flatten() {
            let source = entry.path();
            let target = dest.join(entry.file_name());
            fs::rename(&source, &target)?;
        }
    } else {
        // Move all entries to dest
        for entry in entries {
            let source = entry.path();
            let target = dest.join(entry.file_name());
            fs::rename(&source, &target)?;
        }
    }

    // Clean up temp directory
    fs::remove_dir_all(&temp_dir)?;

    println!("Done");
    Ok(())
}

/// Uninstall cuQuantum from `~/.pecos/deps/cuquantum/`
///
/// # Errors
/// Returns an error if:
/// - Home directory cannot be determined
/// - Directory removal fails
pub fn uninstall_cuquantum() -> Result<()> {
    let cuquantum_dir = get_pecos_cuquantum_dir()?;

    if !cuquantum_dir.exists() {
        println!("cuQuantum is not installed in ~/.pecos/deps/cuquantum/");
        return Ok(());
    }

    println!(
        "Removing cuQuantum installation at: {}",
        cuquantum_dir.display()
    );
    fs::remove_dir_all(&cuquantum_dir)?;
    println!("cuQuantum uninstalled successfully");

    Ok(())
}

/// Ensure cuQuantum is available, installing if needed
///
/// # Errors
/// Returns an error if cuQuantum cannot be found or installed.
pub fn ensure_cuquantum() -> Result<PathBuf> {
    if let Some(path) = super::find_cuquantum() {
        return Ok(path);
    }
    install_cuquantum(false)
}

/// Check if cuQuantum needs to be installed
#[must_use]
pub fn needs_install() -> bool {
    !is_valid_cuquantum_installation(&get_pecos_cuquantum_dir().unwrap_or_default())
        && super::find_cuquantum().is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_platforms_resolve_to_their_checksums() {
        assert_eq!(
            checksum_for("linux-x86_64", 12).unwrap(),
            "77138174ece7a9a9108e2d44b8ea1e1f2e29874544ea940e13c0d607ad15a358"
        );
        assert_eq!(
            checksum_for("linux-sbsa", 13).unwrap(),
            "14a44b09d384c27dd36eae9d9858c90140e6422bbd80bda1041ded96ec3c7223"
        );
    }

    #[test]
    fn unpublished_combinations_are_rejected_with_the_supported_list() {
        // CUDA 11 is not published for this release, and neither is Windows.
        let error = checksum_for("linux-x86_64", 11).unwrap_err().to_string();
        assert!(error.contains("no published cuQuantum"), "{error}");
        assert!(error.contains("linux-x86_64/CUDA 12"), "{error}");
        assert!(checksum_for("windows-x86_64", 12).is_err());
    }

    /// Guards against transcription damage when the table is refreshed by hand: every
    /// entry must be a 64-character lowercase hex digest.
    #[test]
    fn every_checksum_is_well_formed() {
        for (platform, cuda_major, hash) in CUQUANTUM_CHECKSUMS {
            assert_eq!(
                hash.len(),
                64,
                "{platform}/CUDA {cuda_major} digest is not 64 characters"
            );
            assert!(
                hash.chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "{platform}/CUDA {cuda_major} digest is not lowercase hex: {hash}"
            );
        }
    }

    #[test]
    fn the_table_has_no_duplicate_entries() {
        let mut keys: Vec<_> = CUQUANTUM_CHECKSUMS
            .iter()
            .map(|(platform, major, _)| (*platform, *major))
            .collect();
        keys.sort_unstable();
        let before = keys.len();
        keys.dedup();
        assert_eq!(before, keys.len(), "duplicate platform/CUDA entries");
    }
}
