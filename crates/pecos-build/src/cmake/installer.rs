//! Vendored cmake install (Kitware prebuilt binaries).

#![allow(clippy::case_sensitive_file_extension_comparisons)]

use crate::errors::{Error, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const VERSION: &str = super::CMAKE_VERSION;

/// SHA256 checksums for upstream Kitware tarballs we know about.
///
/// Sourced from `cmake-{VERSION}-SHA-256.txt` on the GitHub release page.
const CMAKE_CHECKSUMS: &[(&str, &str)] = &[
    (
        "cmake-3.31.12-linux-x86_64.tar.gz",
        "0dc2e9a6860f06bf10bd8fadc03e35d9eeb4df46e33763a7e480e987758f385c",
    ),
    (
        "cmake-3.31.12-linux-aarch64.tar.gz",
        "83f8fd91d2038a56556e1400390fcfe42f79602940c494f6c6f1cdae7f9e7f40",
    ),
    (
        "cmake-3.31.12-macos-universal.tar.gz",
        "799af7fd545db9bf1b9cfe72f8095880e727a2d4e0df0e3dffc3bc7b95c2d3b0",
    ),
    (
        "cmake-3.31.12-windows-x86_64.zip",
        "0c4baa40f28b3f8225eb3fdf6946c987b4fe901403b4eaf2fbbd9378100aaa0c",
    ),
    (
        "cmake-3.31.12-windows-arm64.zip",
        "e4160c1842dea858ad376ff2ec17587104515b51714eca5963b8bdd798105553",
    ),
];

/// Install cmake to `~/.pecos/deps/cmake-{CMAKE_VERSION}/`.
///
/// # Errors
///
/// Returns an error if the download, checksum verification, or extraction
/// fails, or if the resulting tree doesn't contain a working cmake binary.
pub fn install_cmake(force: bool) -> Result<PathBuf> {
    let cmake_dir = crate::home::get_versioned_dep_path("cmake", VERSION)?;

    if !force && cmake_dir.exists() && is_valid_installation(&cmake_dir) {
        return Err(Error::Config(
            "cmake is already installed. Use --force to reinstall.".into(),
        ));
    }

    if force && cmake_dir.exists() {
        println!("Removing existing cmake installation...");
        fs::remove_dir_all(&cmake_dir)?;
    }

    println!("Installing cmake {VERSION}...");
    println!("This will download ~50MB.");
    println!();

    let (url, archive_name) = get_download_url()?;

    if let Some(parent) = cmake_dir.parent() {
        fs::create_dir_all(parent)?;
    }

    let temp_base = cmake_dir.parent().unwrap_or(&cmake_dir).join("tmp");
    let temp_dir = temp_base.join("cmake");
    fs::create_dir_all(&temp_dir)?;
    let archive_path = temp_dir.join(&archive_name);

    download_and_verify_with_retry(&url, &archive_path, &archive_name)?;

    extract(&archive_path, &cmake_dir, &archive_name)?;
    fs::remove_dir_all(&temp_dir)?;

    if !is_valid_installation(&cmake_dir) {
        return Err(Error::Config(
            "cmake installation completed but verification failed".into(),
        ));
    }

    verify_runtime(&cmake_dir)?;

    println!();
    println!("Installation complete!");
    println!("cmake {VERSION} installed to: {}", cmake_dir.display());

    Ok(cmake_dir)
}

/// Remove the vendored cmake installation.
///
/// # Errors
///
/// Returns an error if removal fails.
pub fn uninstall_cmake() -> Result<()> {
    let cmake_dir = crate::home::get_cmake_dir_path()?;
    if !cmake_dir.exists() {
        println!("cmake is not installed in ~/.pecos/deps/cmake-{VERSION}/");
        return Ok(());
    }
    println!("Removing cmake installation at: {}", cmake_dir.display());
    fs::remove_dir_all(&cmake_dir)?;
    println!("cmake uninstalled successfully");
    Ok(())
}

/// True when the directory contains a working cmake binary in the expected
/// platform-specific location.
#[must_use]
pub fn is_valid_installation(path: &Path) -> bool {
    super::cmake_binary_in(path).is_some()
}

fn get_download_url() -> Result<(String, String)> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    let archive_name = match (os, arch) {
        ("linux", "x86_64") => format!("cmake-{VERSION}-linux-x86_64.tar.gz"),
        ("linux", "aarch64") => format!("cmake-{VERSION}-linux-aarch64.tar.gz"),
        ("macos", _) => format!("cmake-{VERSION}-macos-universal.tar.gz"),
        ("windows", "x86_64") => format!("cmake-{VERSION}-windows-x86_64.zip"),
        ("windows", "aarch64") => format!("cmake-{VERSION}-windows-arm64.zip"),
        _ => {
            return Err(Error::Config(format!(
                "cmake: unsupported platform {os}/{arch}. Install cmake manually from \
                 https://cmake.org/install/ and ensure it is on PATH."
            )));
        }
    };

    let url =
        format!("https://github.com/Kitware/CMake/releases/download/v{VERSION}/{archive_name}");
    Ok((url, archive_name))
}

fn download_and_verify_with_retry(url: &str, dest: &PathBuf, archive_name: &str) -> Result<()> {
    const MAX_RETRIES: u32 = 5;
    const BASE_DELAY_SECS: u64 = 10;

    for attempt in 1..=MAX_RETRIES {
        if attempt > 1 {
            let delay_secs = BASE_DELAY_SECS * (1 << (attempt - 2));
            println!();
            println!("Retry attempt {attempt}/{MAX_RETRIES} (waiting {delay_secs}s)...");
            std::thread::sleep(std::time::Duration::from_secs(delay_secs));
        }

        let _ = fs::remove_file(dest);

        if let Err(e) = download(url, dest) {
            if attempt < MAX_RETRIES {
                eprintln!("Download error: {e}");
                continue;
            }
            return Err(e);
        }

        let file_size = fs::metadata(dest).map_or(0, |m| m.len());
        if file_size == 0 {
            if attempt < MAX_RETRIES {
                eprintln!("Download returned empty file (possible CDN issue)");
                continue;
            }
            return Err(Error::Config(
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

    Err(Error::Config(
        "Download and verification failed after all retries".into(),
    ))
}

fn download(url: &str, dest: &PathBuf) -> Result<()> {
    print!("Downloading cmake... ");
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
            if progress - last_print >= 5.0 {
                print!("\rDownloading cmake... {progress:.0}%");
                io::Write::flush(&mut io::stdout())?;
                last_print = progress;
            }
        }
    }

    println!(
        "\rDownloading cmake... Done ({} MB)",
        downloaded / 1_000_000
    );
    Ok(())
}

fn verify_checksum(file_path: &PathBuf, archive_name: &str) -> Result<()> {
    print!("Verifying checksum... ");
    io::Write::flush(&mut io::stdout())?;

    let data = fs::read(file_path)?;
    let mut hasher = Sha256::new();
    Digest::update(&mut hasher, &data);
    let computed = hasher.finalize().iter().fold(String::new(), |mut s, b| {
        use std::fmt::Write;
        write!(s, "{b:02x}").unwrap();
        s
    });

    let expected = CMAKE_CHECKSUMS
        .iter()
        .find(|(name, _)| *name == archive_name)
        .map(|(_, hash)| *hash);

    match expected {
        Some(hash) if !hash.is_empty() => {
            if computed == hash {
                println!("OK");
                Ok(())
            } else {
                println!("FAILED");
                Err(Error::Sha256Mismatch {
                    expected: hash.to_string(),
                    actual: computed,
                })
            }
        }
        _ => {
            println!("Skipped (checksum not available for {archive_name})");
            Ok(())
        }
    }
}

fn extract(archive: &Path, dest: &Path, archive_name: &str) -> Result<()> {
    print!("Extracting cmake... ");
    io::Write::flush(&mut io::stdout())?;

    if archive_name.ends_with(".tar.gz") {
        extract_tar_gz(archive, dest, archive_name)?;
    } else if archive_name.ends_with(".zip") {
        extract_zip(archive, dest, archive_name)?;
    } else {
        return Err(Error::Archive(format!(
            "Unsupported cmake archive format: {archive_name}"
        )));
    }

    println!("Done");
    Ok(())
}

fn extract_tar_gz(archive: &Path, dest: &Path, archive_name: &str) -> Result<()> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    let file = fs::File::open(archive)?;
    let decompressor = GzDecoder::new(file);
    let mut tar_archive = Archive::new(decompressor);

    let extract_to = dest
        .parent()
        .ok_or_else(|| Error::Archive("Invalid destination path".into()))?;
    tar_archive.unpack(extract_to)?;

    // Upstream tarballs unpack to "cmake-{VERSION}-{platform}/" — rename to our
    // canonical name "cmake-{VERSION}/".
    let extracted_dir_name = archive_name.strip_suffix(".tar.gz").unwrap_or(archive_name);
    let extracted_dir = extract_to.join(extracted_dir_name);
    if extracted_dir.exists() && !dest.exists() {
        fs::rename(&extracted_dir, dest)?;
    }
    Ok(())
}

fn extract_zip(archive: &Path, dest: &Path, archive_name: &str) -> Result<()> {
    use zip::ZipArchive;

    let file = fs::File::open(archive)?;
    let mut zip = ZipArchive::new(file).map_err(|e| Error::Archive(e.to_string()))?;

    let extract_to = dest
        .parent()
        .ok_or_else(|| Error::Archive("Invalid destination path".into()))?;

    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| Error::Archive(e.to_string()))?;
        let Some(entry_path) = entry.enclosed_name() else {
            continue;
        };
        let out_path = extract_to.join(entry_path);
        if entry.is_dir() {
            fs::create_dir_all(&out_path)?;
        } else {
            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut out_file = fs::File::create(&out_path)?;
            io::copy(&mut entry, &mut out_file)?;
        }
    }

    let extracted_dir_name = archive_name.strip_suffix(".zip").unwrap_or(archive_name);
    let extracted_dir = extract_to.join(extracted_dir_name);
    if extracted_dir.exists() && !dest.exists() {
        fs::rename(&extracted_dir, dest)?;
    }
    Ok(())
}

fn verify_runtime(cmake_dir: &Path) -> Result<()> {
    print!("Verifying cmake runtime... ");
    io::Write::flush(&mut io::stdout())?;

    let Some(cmake_bin) = super::cmake_binary_in(cmake_dir) else {
        println!("FAILED");
        return Err(Error::Config(
            "cmake binary missing from extracted tree".into(),
        ));
    };

    let output = std::process::Command::new(&cmake_bin)
        .arg("--version")
        .output()
        .map_err(|e| Error::Config(format!("Failed to execute cmake: {e}")))?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let version_line = stdout.lines().next().unwrap_or("").trim();
        println!("OK ({version_line})");
        Ok(())
    } else {
        println!("FAILED");
        Err(Error::Config(
            "cmake --version exited with non-zero status".into(),
        ))
    }
}
