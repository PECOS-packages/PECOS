//! cuTensor library management for PECOS
//!
//! cuTensor is a required runtime dependency of cuTensorNet (part of cuQuantum).
//! This module handles finding and installing cuTensor to `~/.pecos/deps/`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::errors::{Error, Result};
use crate::home::{get_cache_dir, get_deps_dir};

/// cuTensor version to install (major.minor.patch.build)
pub const CUTENSOR_VERSION: &str = "2.4.1.4";

/// SHA256 checksums for the [`CUTENSOR_VERSION`] archives, by platform key and CUDA major.
///
/// NVIDIA publishes these in the release manifest beside the archives themselves. To
/// refresh when bumping [`CUTENSOR_VERSION`], read the `sha256` fields from
/// `https://developer.download.nvidia.com/compute/cutensor/redist/redistrib_<release>.json`
/// (the release label drops the build component, so `2.4.1.4` is published as
/// `redistrib_2.4.1.json`). Take the values from that manifest, never from a local
/// download, so that a substituted archive cannot certify itself.
const CUTENSOR_CHECKSUMS: &[(&str, u32, &str)] = &[
    (
        "linux-x86_64",
        12,
        "032904fb8bba341e24aa45a8cc7b5afc63e4c28e22474530ccc97cfa546d0442",
    ),
    (
        "linux-x86_64",
        13,
        "21fb0a9a3b7b6663223667b587f2ab1c54e4a61c975fea0f9d4f56d9c33f31fe",
    ),
    (
        "linux-sbsa",
        12,
        "afcf1bd3a50b729bcd5d1ddb0a3e90ca2631d7048d51bdeafe49c650e162ebc1",
    ),
    (
        "linux-sbsa",
        13,
        "9baffd3658b7f4da2d2f94d23c3acddb6d12c62997d3da39a774a058fef04aa5",
    ),
];

/// Looks up the published checksum for a platform and CUDA major version.
fn checksum_for(platform: &str, cuda_major: u32) -> Result<&'static str> {
    CUTENSOR_CHECKSUMS
        .iter()
        .find(|(key, major, _)| *key == platform && *major == cuda_major)
        .map(|(_, _, hash)| *hash)
        .ok_or_else(|| {
            Error::CuQuantum(format!(
                "no published cuTensor {CUTENSOR_VERSION} archive for {platform} with CUDA \
                 {cuda_major}; supported combinations are {}",
                CUTENSOR_CHECKSUMS
                    .iter()
                    .map(|(key, major, _)| format!("{key}/CUDA {major}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
}

/// Computes the SHA256 of a file, hashing incrementally.
///
/// These archives are roughly half a gigabyte and verification runs on every install, so
/// reading one wholly into memory would allocate that much each time.
fn file_sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};

    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let read = std::io::Read::read(&mut file, &mut buffer)?;
        if read == 0 {
            break;
        }
        Digest::update(&mut hasher, &buffer[..read]);
    }
    Ok(hasher.finalize().iter().fold(String::new(), |mut s, b| {
        use std::fmt::Write;
        write!(s, "{b:02x}").unwrap();
        s
    }))
}

/// Verifies an archive against its published digest.
fn verify_archive(path: &Path, expected: &str) -> Result<()> {
    let computed = file_sha256(path)?;
    if computed == expected {
        Ok(())
    } else {
        Err(Error::Sha256Mismatch {
            expected: expected.to_string(),
            actual: computed,
        })
    }
}

/// Get the expected cuTensor directory name in deps
fn dep_dir_name() -> String {
    format!("cutensor-{CUTENSOR_VERSION}")
}

/// Find cuTensor installation
///
/// Search order:
/// 1. `~/.pecos/deps/cutensor-<version>/`
/// 2. System paths
#[must_use]
pub fn find_cutensor() -> Option<PathBuf> {
    // 1. Check ~/.pecos/deps/
    if let Ok(deps_dir) = get_deps_dir() {
        let path = deps_dir.join(dep_dir_name());
        if is_valid_cutensor(&path) {
            return Some(path);
        }
    }

    // 2. Check alongside cuQuantum installation
    if let Some(cuquantum_path) = crate::cuquantum::find_cuquantum()
        && has_cutensor_lib(&cuquantum_path)
    {
        return Some(cuquantum_path);
    }

    None
}

/// Check if a path contains cutensor libraries
#[must_use]
fn is_valid_cutensor(path: &Path) -> bool {
    has_cutensor_lib(path)
}

/// Check if libcutensor exists in a lib or lib64 subdirectory
#[must_use]
fn has_cutensor_lib(path: &Path) -> bool {
    for lib_dir in &["lib", "lib64"] {
        let lib = path.join(lib_dir).join("libcutensor.so");
        if lib.exists() {
            return true;
        }
        let lib_versioned = path.join(lib_dir).join("libcutensor.so.2");
        if lib_versioned.exists() {
            return true;
        }
    }
    false
}

/// Get the library directory within a cuTensor installation
#[must_use]
pub fn get_lib_dir(cutensor_path: &Path) -> Option<PathBuf> {
    let lib64 = cutensor_path.join("lib64");
    if lib64.exists() {
        return Some(lib64);
    }
    let lib = cutensor_path.join("lib");
    if lib.exists() {
        return Some(lib);
    }
    None
}

/// Install cuTensor to `~/.pecos/deps/cutensor-<version>/`
///
/// Downloads the cuTensor redistributable from NVIDIA and extracts it.
///
/// # Errors
///
/// Returns an error if download or extraction fails.
pub fn install_cutensor(force: bool) -> Result<PathBuf> {
    let deps_dir = get_deps_dir()?;
    let dest = deps_dir.join(dep_dir_name());

    if !force && is_valid_cutensor(&dest) {
        return Ok(dest);
    }

    if force && dest.exists() {
        fs::remove_dir_all(&dest)?;
    }

    let (url, filename, expected_sha256) = get_download_info()?;
    let cache_dir = get_cache_dir()?;
    let archive_path = cache_dir.join(&filename);

    // Verify on every install, cached or freshly downloaded: this archive is extracted
    // into ~/.pecos and loaded at runtime, and the cache persists between runs, so a
    // truncated or tampered file must not be trusted because an earlier run left it there.
    // A cached archive that fails is discarded and re-fetched once so a corrupt file
    // cannot wedge every later build.
    if archive_path.exists() {
        println!(
            "cargo:warning=Using cached cuTensor download: {}",
            archive_path.display()
        );
        if let Err(error) = verify_archive(&archive_path, expected_sha256) {
            println!(
                "cargo:warning=Cached cuTensor archive failed verification ({error}); refetching"
            );
            // A concurrent build may have already removed or replaced the bad cache entry
            // between our verify and here; a missing file is exactly the state we wanted, so
            // do not abort the refetch over it.
            if let Err(remove_error) = fs::remove_file(&archive_path)
                && remove_error.kind() != std::io::ErrorKind::NotFound
            {
                return Err(remove_error.into());
            }
            fetch_and_verify(&url, &archive_path, expected_sha256)?;
        }
    } else {
        fetch_and_verify(&url, &archive_path, expected_sha256)?;
    }

    // Extract with --strip-components=1 into dest
    fs::create_dir_all(&dest)?;

    let status = Command::new("tar")
        .arg("-xf")
        .arg(&archive_path)
        .arg("-C")
        .arg(&dest)
        .arg("--strip-components=1")
        .status()
        .map_err(|e| Error::Archive(format!("Failed to run tar: {e}")))?;

    if !status.success() {
        return Err(Error::Archive("cuTensor extraction failed".into()));
    }

    if !is_valid_cutensor(&dest) {
        return Err(Error::Archive(
            "cuTensor extraction succeeded but no libraries found".into(),
        ));
    }

    println!(
        "cargo:warning=cuTensor {CUTENSOR_VERSION} installed to: {}",
        dest.display()
    );

    Ok(dest)
}

/// Ensure cuTensor is available, installing if needed
///
/// # Errors
///
/// Returns an error if cuTensor cannot be found or installed.
pub fn ensure_cutensor() -> Result<PathBuf> {
    if let Some(path) = find_cutensor() {
        return Ok(path);
    }
    install_cutensor(false)
}

/// Detect CUDA major version
fn detect_cuda_major() -> u32 {
    if let Some(cuda_path) = crate::cuda::find_cuda()
        && let Ok(version) = crate::cuda::get_cuda_version(&cuda_path)
        && let Some(major) = version.split('.').next()
        && let Ok(v) = major.parse::<u32>()
    {
        return v;
    }
    12
}

/// Downloads to a private `.part` sibling, verifies it, and only then moves it into place.
///
/// Nothing is published at the cache path until its checksum matches, so an interrupted
/// download or a killed process cannot leave a file a later build mistakes for verified.
///
/// The partial name carries this process's id. Concurrent builds share the cache directory
/// (one per worktree is a normal workflow), and a shared partial name would let one build
/// truncate another's download between its verify and its rename -- so the file that was
/// verified would not be the file renamed into place. A per-process partial keeps each
/// build's verify-then-rename operating on bytes no other build can touch.
fn fetch_and_verify(url: &str, archive_path: &Path, expected_sha256: &str) -> Result<()> {
    println!("cargo:warning=Downloading cuTensor {CUTENSOR_VERSION}...");
    let partial_path = archive_path.with_extension(format!("part.{}", std::process::id()));
    download(url, &partial_path)?;

    if let Err(error) = verify_archive(&partial_path, expected_sha256) {
        let _ = fs::remove_file(&partial_path);
        return Err(error);
    }

    fs::rename(&partial_path, archive_path)?;
    Ok(())
}

/// Get platform-specific download URL, filename, and published checksum
fn get_download_info() -> Result<(String, String, &'static str)> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let cuda_major = detect_cuda_major();

    // cuTensor 2.3+ archives include _cuda12 or _cuda13 suffix
    match (os, arch) {
        ("linux", "x86_64") => Ok((
            format!(
                "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-{CUTENSOR_VERSION}_cuda{cuda_major}-archive.tar.xz"
            ),
            format!("libcutensor-linux-x86_64-{CUTENSOR_VERSION}_cuda{cuda_major}-archive.tar.xz"),
            checksum_for("linux-x86_64", cuda_major)?,
        )),
        ("linux", "aarch64") => Ok((
            format!(
                "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-sbsa/libcutensor-linux-sbsa-{CUTENSOR_VERSION}_cuda{cuda_major}-archive.tar.xz"
            ),
            format!("libcutensor-linux-sbsa-{CUTENSOR_VERSION}_cuda{cuda_major}-archive.tar.xz"),
            checksum_for("linux-sbsa", cuda_major)?,
        )),
        _ => Err(Error::CuQuantum(format!(
            "cuTensor is not available for {os}/{arch}"
        ))),
    }
}

/// Download a file from a URL, streaming to disk
fn download(url: &str, dest: &Path) -> Result<()> {
    crate::download::ensure_crypto_provider();
    let mut response = reqwest::blocking::get(url).map_err(|e| Error::Http(e.to_string()))?;

    if !response.status().is_success() {
        return Err(Error::Http(format!(
            "cuTensor download failed with status: {}",
            response.status()
        )));
    }

    let mut file = fs::File::create(dest)?;
    let bytes_copied = std::io::copy(&mut response, &mut file)
        .map_err(|e| Error::Http(format!("Failed to write download to disk: {e}")))?;

    println!(
        "cargo:warning=Downloaded cuTensor ({} MB)",
        bytes_copied / 1_000_000
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `checksum_for` selects the row matching BOTH the platform and the CUDA major.
    ///
    /// This pins the lookup, not the digest values: the literals here are the same
    /// constants the table holds, so this cannot catch a wrong digit in the table -- only a
    /// lookup that returns the wrong row (for example, ignoring the CUDA major). The table
    /// values themselves were checked against NVIDIA's manifest by hand; see the note on
    /// [`CUTENSOR_CHECKSUMS`] for how to re-verify them on a version bump.
    #[test]
    fn checksum_for_selects_the_row_matching_platform_and_cuda_major() {
        assert_eq!(
            checksum_for("linux-x86_64", 12).unwrap(),
            "032904fb8bba341e24aa45a8cc7b5afc63e4c28e22474530ccc97cfa546d0442"
        );
        assert_eq!(
            checksum_for("linux-sbsa", 13).unwrap(),
            "9baffd3658b7f4da2d2f94d23c3acddb6d12c62997d3da39a774a058fef04aa5"
        );
        // The x86-64 and sbsa CUDA-12 digests differ, so a lookup that dropped the platform
        // would return one of these where the other is expected.
        assert_ne!(
            checksum_for("linux-x86_64", 12).unwrap(),
            checksum_for("linux-sbsa", 12).unwrap()
        );
    }

    #[test]
    fn unpublished_combinations_are_rejected_with_the_supported_list() {
        let error = checksum_for("linux-x86_64", 11).unwrap_err().to_string();
        assert!(error.contains("no published cuTensor"), "{error}");
        assert!(error.contains("linux-x86_64/CUDA 12"), "{error}");
    }

    /// Catches a digest whose SHAPE is wrong (not 64 lowercase-hex characters). It cannot
    /// catch a wrong-but-still-hex value; that is the manual manifest check's job.
    #[test]
    fn every_checksum_is_well_formed() {
        for (platform, cuda_major, hash) in CUTENSOR_CHECKSUMS {
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
        let mut keys: Vec<_> = CUTENSOR_CHECKSUMS
            .iter()
            .map(|(platform, major, _)| (*platform, *major))
            .collect();
        keys.sort_unstable();
        let before = keys.len();
        keys.dedup();
        assert_eq!(before, keys.len(), "duplicate platform/CUDA entries");
    }

    #[test]
    fn digests_match_and_reject_content() {
        // Pins verify_archive against a known-answer digest rather than only exercising
        // the table: a hash function wired up wrongly would still satisfy the table tests.
        let dir = std::env::temp_dir().join("pecos-cutensor-digest-test");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sample");
        fs::write(&path, b"abc").unwrap();
        let sha256_of_abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(file_sha256(&path).unwrap(), sha256_of_abc);
        assert!(verify_archive(&path, sha256_of_abc).is_ok());
        assert!(matches!(
            verify_archive(&path, &"0".repeat(64)),
            Err(Error::Sha256Mismatch { .. })
        ));
        fs::remove_dir_all(&dir).ok();
    }
}
