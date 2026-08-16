//! CUDA Toolkit installation functionality
//!
//! Downloads and installs CUDA Toolkit to `~/.pecos/deps/cuda/`

#![allow(clippy::case_sensitive_file_extension_comparisons)]

use crate::errors::{Error, Result};
use crate::extract::contained_entry_path;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use super::{CUDA_VERSION, get_pecos_cuda_dir, is_valid_cuda_installation};

const CUDA_REDIST_BASE: &str = "https://developer.download.nvidia.com/compute/cuda/redist";

/// Components installed for CUDA 12.6.3, in installation order.
///
/// NVIDIA versions redistributable components independently of the CUDA release, so keep
/// each component's published version rather than deriving it from [`CUDA_VERSION`].
const CUDA_COMPONENTS: &[(&str, &str)] = &[
    ("cuda_nvcc", "12.6.85"),
    ("cuda_cudart", "12.6.77"),
    ("libcublas", "12.6.4.1"),
];

struct CudaArchive {
    component: &'static str,
    platform: &'static str,
    filename: &'static str,
    sha256: &'static str,
}

/// CUDA redistributable archives keyed by component and platform.
///
/// NVIDIA published these values in
/// `https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json`.
/// To refresh them, read the `cuda_nvcc`, `cuda_cudart`, and `libcublas` keys from the new
/// manifest. Each has its own `version` and per-platform `relative_path` and `sha256`;
/// preserve the component-specific version in the filename and take the digest from the
/// manifest, never from a local download.
const CUDA_ARCHIVES: &[CudaArchive] = &[
    CudaArchive {
        component: "cuda_nvcc",
        platform: "linux-x86_64",
        filename: "cuda_nvcc-linux-x86_64-12.6.85-archive.tar.xz",
        sha256: "840deff234d9bef20d6856439c49881cb4f29423b214f9ecd2fa59b7ac323817",
    },
    CudaArchive {
        component: "cuda_nvcc",
        platform: "linux-sbsa",
        filename: "cuda_nvcc-linux-sbsa-12.6.85-archive.tar.xz",
        sha256: "1b834df41cb071884f33b1e4ffc185e4799975057baca57d80ba7c4591e67950",
    },
    CudaArchive {
        component: "cuda_nvcc",
        platform: "windows-x86_64",
        filename: "cuda_nvcc-windows-x86_64-12.6.85-archive.zip",
        sha256: "3fb9f76b87c37d02f947354be89b718ad5f2c76b6ab47995265bfa3a068a5e14",
    },
    CudaArchive {
        component: "cuda_cudart",
        platform: "linux-x86_64",
        filename: "cuda_cudart-linux-x86_64-12.6.77-archive.tar.xz",
        sha256: "f74689258a60fd9c5bdfa7679458527a55e22442691ba678dcfaeffbf4391ef9",
    },
    CudaArchive {
        component: "cuda_cudart",
        platform: "linux-sbsa",
        filename: "cuda_cudart-linux-sbsa-12.6.77-archive.tar.xz",
        sha256: "c73c8e5bfe8fcd7468d012c9eebff15063005a3bba44423d541d573dc058de58",
    },
    CudaArchive {
        component: "cuda_cudart",
        platform: "windows-x86_64",
        filename: "cuda_cudart-windows-x86_64-12.6.77-archive.zip",
        sha256: "7a313bc0c93b1a50bb03aa9783a199ae70c3b66e2d8084da65e8254a8577b925",
    },
    CudaArchive {
        component: "libcublas",
        platform: "linux-x86_64",
        filename: "libcublas-linux-x86_64-12.6.4.1-archive.tar.xz",
        sha256: "ec682bac6387f9cdfd0c20b25a16cd6ed0b8b3b7ff42be9eaeb41828e3a72572",
    },
    CudaArchive {
        component: "libcublas",
        platform: "linux-sbsa",
        filename: "libcublas-linux-sbsa-12.6.4.1-archive.tar.xz",
        sha256: "84668dcb2159f9efd912a66ed5afe5d6533b72a81bbabc98b26ac7ac7a36105a",
    },
    CudaArchive {
        component: "libcublas",
        platform: "windows-x86_64",
        filename: "libcublas-windows-x86_64-12.6.4.1-archive.zip",
        sha256: "1a87ec80f8c0e5a39badc87010d479930c5b63abd788b3a05bd688a5980a3d07",
    },
];

fn platform_key() -> Result<&'static str> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    match (os, arch) {
        ("linux", "x86_64") => Ok("linux-x86_64"),
        ("linux", "aarch64") => Ok("linux-sbsa"),
        ("windows", "x86_64") => Ok("windows-x86_64"),
        ("macos", _) => Err(Error::Cuda(
            "CUDA is not supported on macOS (deprecated by NVIDIA since macOS 10.14)".into(),
        )),
        _ => Err(Error::Cuda(format!("Unsupported platform: {os}/{arch}"))),
    }
}

fn archive_for(component: &str, platform: &str) -> Result<&'static CudaArchive> {
    CUDA_ARCHIVES
        .iter()
        .find(|archive| archive.component == component && archive.platform == platform)
        .ok_or_else(|| {
            Error::Cuda(format!(
                "No CUDA redistributable archive for {component} on {platform}"
            ))
        })
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
/// - Download, verification, extraction, or merging fails
/// - Installation verification fails
pub fn install_cuda(force: bool) -> Result<PathBuf> {
    let cuda_dir = crate::home::get_versioned_dep_path("cuda", CUDA_VERSION)?;

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

    let platform = platform_key()?;

    println!("Installing CUDA Toolkit {CUDA_VERSION}...");
    println!(
        "This will download ~574MB and may take several minutes depending on your connection."
    );
    println!();

    // Create cache directory
    let cuda_parent = cuda_dir
        .parent()
        .ok_or_else(|| Error::Cuda("Invalid CUDA directory".into()))?;
    let cache_dir = cuda_parent.join("cache");
    fs::create_dir_all(&cache_dir)?;

    let extraction_root = cuda_parent
        .join("tmp")
        .join(format!("cuda_extract.{}", std::process::id()));
    if let Err(error) = fs::remove_dir_all(&extraction_root)
        && error.kind() != io::ErrorKind::NotFound
    {
        return Err(error.into());
    }
    fs::create_dir_all(&extraction_root)?;

    let install_result = install_components(platform, &cache_dir, &extraction_root, &cuda_dir);
    let cleanup_result = fs::remove_dir_all(&extraction_root);
    install_result?;
    cleanup_result?;

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

fn install_components(
    platform: &str,
    cache_dir: &Path,
    extraction_root: &Path,
    cuda_dir: &Path,
) -> Result<()> {
    for (component, version) in CUDA_COMPONENTS {
        let archive = archive_for(component, platform)?;
        debug_assert!(archive.filename.contains(version));

        let url = format!(
            "{CUDA_REDIST_BASE}/{component}/{platform}/{}",
            archive.filename
        );
        let archive_path = cache_dir.join(archive.filename);

        // Verify on every install, whether cached or freshly downloaded. A cached archive
        // that fails is discarded and re-fetched once so corruption cannot wedge later runs.
        if archive_path.exists() {
            println!("Using cached download: {}", archive_path.display());
            if let Err(error) = verify_archive(&archive_path, archive.sha256) {
                println!(
                    "Cached CUDA component {} failed verification ({error}); refetching",
                    archive.filename
                );
                // Another process may have removed the bad entry after our verification.
                if let Err(remove_error) = fs::remove_file(&archive_path)
                    && remove_error.kind() != io::ErrorKind::NotFound
                {
                    return Err(remove_error.into());
                }
                fetch_and_verify(&url, &archive_path, archive.sha256)?;
            }
        } else {
            fetch_and_verify(&url, &archive_path, archive.sha256)?;
        }

        let component_root = extraction_root.join(component);
        fs::create_dir_all(&component_root)?;
        let extracted_dir = extract_component_archive(&archive_path, &component_root)?;
        merge_component_dir(&extracted_dir, cuda_dir)?;
    }

    Ok(())
}

/// Downloads to a private `.part` sibling, verifies it, and only then moves it into place.
///
/// The process id prevents concurrent installs from modifying the bytes between verification
/// and rename. A failed verification removes the partial file and never publishes it.
fn fetch_and_verify(url: &str, archive_path: &Path, expected_sha256: &str) -> Result<()> {
    let filename = archive_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("CUDA component");
    println!("Downloading {filename}...");
    let partial_path = archive_path.with_extension(format!("part.{}", std::process::id()));
    if let Err(error) = download_cuda(url, &partial_path) {
        // A download interrupted midway leaves a partial file; do not leave it behind for
        // a later run to trip over (verification would reject it, but the leak is untidy).
        let _ = fs::remove_file(&partial_path);
        return Err(error);
    }

    if let Err(error) = verify_archive(&partial_path, expected_sha256) {
        let _ = fs::remove_file(&partial_path);
        return Err(error);
    }

    fs::rename(&partial_path, archive_path)?;
    Ok(())
}

/// Download a CUDA component, streaming it to disk.
fn download_cuda(url: &str, dest: &Path) -> Result<()> {
    print!("Downloading CUDA component... ");
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
    let mut buffer = vec![0u8; 64 * 1024];

    loop {
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
                print!("\rDownloading CUDA component... {progress:.0}%");
                io::stdout().flush()?;
                last_print = progress;
            }
        }
    }

    println!(
        "\rDownloading CUDA component... Done ({} MB)",
        downloaded / 1_000_000
    );
    Ok(())
}

/// Computes a file's SHA256 incrementally so large archives are not read into memory.
fn file_sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};

    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let read = io::Read::read(&mut file, &mut buffer)?;
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

fn verify_archive(path: &Path, expected: &str) -> Result<()> {
    let actual = file_sha256(path)?;
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Sha256Mismatch {
            expected: expected.to_string(),
            actual,
        })
    }
}

fn extract_component_archive(archive: &Path, dest: &Path) -> Result<PathBuf> {
    let filename = archive
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::Archive("Invalid archive path".into()))?;

    println!("Extracting {filename}...");
    if filename.ends_with(".tar.xz") {
        extract_tar_xz(archive, dest)?;
    } else if filename.ends_with(".zip") {
        extract_zip(archive, dest)?;
    } else {
        return Err(Error::Archive(format!(
            "Unsupported CUDA component archive format: {filename}"
        )));
    }

    let expected_name = filename
        .strip_suffix(".tar.xz")
        .or_else(|| filename.strip_suffix(".zip"))
        .ok_or_else(|| Error::Archive(format!("Invalid CUDA archive filename: {filename}")))?;
    validate_extracted_layout(dest, expected_name)
}

fn extract_tar_xz(archive: &Path, dest: &Path) -> Result<()> {
    let file = fs::File::open(archive)?;
    let decoder = xz2::read::XzDecoder::new(file);
    let mut tar_archive = tar::Archive::new(decoder);
    tar_archive.unpack(dest)?;
    Ok(())
}

fn extract_zip(archive: &Path, dest: &Path) -> Result<()> {
    let file = fs::File::open(archive)?;
    let mut zip = zip::ZipArchive::new(file).map_err(|error| Error::Archive(error.to_string()))?;

    for index in 0..zip.len() {
        let mut entry = zip
            .by_index(index)
            .map_err(|error| Error::Archive(error.to_string()))?;
        let Some(entry_path) = entry.enclosed_name() else {
            continue;
        };
        // `enclosed_name` balances `..` lexically but accepts Windows-normalized escapes
        // and device names, so every accepted entry also goes through the shared guard.
        let out_path = contained_entry_path(dest, &entry_path.to_string_lossy())?;
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

    Ok(())
}

fn validate_extracted_layout(root: &Path, expected_name: &str) -> Result<PathBuf> {
    let entries = fs::read_dir(root)?.collect::<std::result::Result<Vec<_>, _>>()?;
    if entries.len() != 1 {
        return Err(Error::Archive(format!(
            "CUDA component archive must contain one top-level directory named {expected_name}; found {} entries",
            entries.len()
        )));
    }

    let entry = &entries[0];
    if !entry.file_type()?.is_dir() || entry.file_name() != expected_name {
        return Err(Error::Archive(format!(
            "CUDA component archive top-level entry must be the directory {expected_name}; found {}",
            entry.path().display()
        )));
    }

    Ok(entry.path())
}

/// Creates a symlink at `link` pointing at `target`, on either platform.
#[cfg(unix)]
fn symlink_file(target: &Path, link: &Path) -> Result<()> {
    std::os::unix::fs::symlink(target, link)?;
    Ok(())
}

/// Creates a symlink at `link` pointing at `target`, on either platform.
///
/// CUDA `lib/` symlink targets are files, so the file variant is correct here.
#[cfg(windows)]
fn symlink_file(target: &Path, link: &Path) -> Result<()> {
    std::os::windows::fs::symlink_file(target, link)?;
    Ok(())
}

/// True if a symlink written at `link_path` with contents `link_target` resolves to a path
/// inside `root`.
///
/// The target is resolved lexically against the link's parent directory (the link does not
/// exist yet, so the filesystem cannot resolve it): absolute targets and targets that climb
/// out of `root` with `..` are rejected. `root` is an absolute, `..`-free path, so lexical
/// resolution is sufficient here.
fn symlink_target_stays_within(root: &Path, link_path: &Path, link_target: &Path) -> bool {
    if link_target.is_absolute() {
        return false;
    }
    let Some(parent) = link_path.parent() else {
        return false;
    };
    let mut resolved = parent.to_path_buf();
    for component in link_target.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !resolved.pop() {
                    return false;
                }
            }
            std::path::Component::Normal(name) => resolved.push(name),
            std::path::Component::RootDir | std::path::Component::Prefix(_) => return false,
        }
    }
    resolved.starts_with(root)
}

/// Recursively merges a component's top-level children into the CUDA installation.
///
/// Uses the non-following [`std::fs::DirEntry::file_type`] to classify each entry, and
/// recreates symlinks as symlinks rather than copying through them. CUDA `lib/` trees ship
/// versioned shared-object symlink chains (`libcublas.so -> libcublas.so.12 ->
/// libcublas.so.12.6.4.1`); copying through them with a following `is_dir`/`fs::copy` would
/// both triple the ~523MB payload on disk and follow a symlinked directory out of the
/// extraction tree. Recreating the link preserves the layout NVIDIA intended and keeps the
/// merge within the extracted component.
fn merge_component_dir(component_dir: &Path, cuda_dir: &Path) -> Result<()> {
    fs::create_dir_all(cuda_dir)?;
    for entry in fs::read_dir(component_dir)? {
        let entry = entry?;
        let source = entry.path();
        let target = cuda_dir.join(entry.file_name());
        let file_type = entry.file_type()?;

        if file_type.is_symlink() {
            // Redist archives only carry relative symlinks pointing beside the link; recreate
            // it verbatim so `.so` version resolution keeps working without duplicating bytes.
            let link_target = fs::read_link(&source)?;
            // Defense in depth against a symlink that resolves outside the install tree.
            // The archive is SHA256-verified, so this can only fire on an NVIDIA-authored
            // link, but a link whose target is absolute or climbs out with `..` would let a
            // later reader escape `cuda_dir`; reject it rather than recreate it.
            if !symlink_target_stays_within(cuda_dir, &target, &link_target) {
                return Err(Error::Archive(format!(
                    "CUDA component symlink {} points outside the installation: {}",
                    target.display(),
                    link_target.display()
                )));
            }
            if target.symlink_metadata().is_ok() {
                fs::remove_file(&target)?;
            }
            symlink_file(&link_target, &target)?;
        } else if file_type.is_dir() {
            fs::create_dir_all(&target)?;
            merge_component_dir(&source, &target)?;
        } else {
            fs::copy(&source, &target)?;
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_checksum_is_well_formed() {
        for archive in CUDA_ARCHIVES {
            assert_eq!(
                archive.sha256.len(),
                64,
                "{}/{} digest is not 64 characters",
                archive.component,
                archive.platform
            );
            assert!(
                archive
                    .sha256
                    .chars()
                    .all(|character| character.is_ascii_digit() || ('a'..='f').contains(&character)),
                "{}/{} digest is not lowercase hex: {}",
                archive.component,
                archive.platform,
                archive.sha256
            );
        }
    }

    #[test]
    fn the_archive_table_has_no_duplicate_entries() {
        let mut keys: Vec<_> = CUDA_ARCHIVES
            .iter()
            .map(|archive| (archive.component, archive.platform))
            .collect();
        keys.sort_unstable();
        let before = keys.len();
        keys.dedup();
        assert_eq!(before, keys.len(), "duplicate component/platform entries");
    }

    #[test]
    fn streaming_hash_matches_known_answer() {
        let temp = tempfile::tempdir().expect("tempdir");
        let sample = temp.path().join("sample");
        fs::write(&sample, b"abc").expect("write sample");

        assert_eq!(
            file_sha256(&sample).expect("hash sample"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn component_trees_merge_into_one_cuda_installation() {
        let temp = tempfile::tempdir().expect("tempdir");
        let component_a = temp.path().join("comp-a-archive");
        let component_b = temp.path().join("comp-b-archive");
        let cuda_dir = temp.path().join("cuda");

        fs::create_dir_all(component_a.join("bin")).expect("create bin fixture");
        fs::write(component_a.join("bin/nvcc"), b"nvcc").expect("write nvcc fixture");
        fs::create_dir_all(component_b.join("include")).expect("create include fixture");
        fs::write(
            component_b.join("include/cuda_runtime.h"),
            b"runtime header",
        )
        .expect("write header fixture");

        merge_component_dir(&component_a, &cuda_dir).expect("merge first component");
        merge_component_dir(&component_b, &cuda_dir).expect("merge second component");

        assert!(cuda_dir.join("bin/nvcc").exists());
        assert!(cuda_dir.join("include/cuda_runtime.h").exists());
    }

    #[test]
    fn symlink_target_containment_accepts_sonames_and_rejects_escapes() {
        let root = Path::new("/out/cuda");
        let link = root.join("lib/libcublas.so");
        // Legitimate relative soname links pointing beside themselves are accepted.
        assert!(symlink_target_stays_within(
            root,
            &link,
            Path::new("libcublas.so.12")
        ));
        assert!(symlink_target_stays_within(
            root,
            &link,
            Path::new("./libcublas.so.12")
        ));
        // Absolute and climbing-out targets are rejected.
        assert!(!symlink_target_stays_within(
            root,
            &link,
            Path::new("/etc/passwd")
        ));
        assert!(!symlink_target_stays_within(
            root,
            &link,
            Path::new("../../../etc/passwd")
        ));
        // A `..` that stays inside is fine (lib/../include/foo -> /out/cuda/include/foo).
        assert!(symlink_target_stays_within(
            root,
            &link,
            Path::new("../include/cuda_runtime.h")
        ));
        // The prefix trap: a sister directory that shares a string prefix with the install
        // dir must be rejected. `../../cuda-evil/x` resolves to `/out/cuda-evil/x`, which a
        // naive string `starts_with` would accept; the component-wise check must not.
        assert!(!symlink_target_stays_within(
            root,
            &link,
            Path::new("../../cuda-evil/libcublas.so")
        ));
        // Popping above the root is rejected even when the tail re-enters a same-named dir.
        assert!(!symlink_target_stays_within(
            root,
            &link,
            Path::new("../lib/../../cuda-evil/x")
        ));
    }

    /// The redist `lib/` trees ship versioned soname symlink chains. Merging must recreate
    /// them as symlinks, not copy through them into duplicate full-size files.
    #[cfg(unix)]
    #[test]
    fn merge_preserves_shared_object_symlink_chains() {
        let temp = tempfile::tempdir().expect("tempdir");
        let component = temp.path().join("comp-archive");
        let cuda_dir = temp.path().join("cuda");
        let lib = component.join("lib");
        fs::create_dir_all(&lib).expect("create lib fixture");

        // libcublas.so -> libcublas.so.12 -> libcublas.so.12.6.4.1 (the real file)
        fs::write(lib.join("libcublas.so.12.6.4.1"), vec![0u8; 4096]).expect("write real so");
        std::os::unix::fs::symlink("libcublas.so.12.6.4.1", lib.join("libcublas.so.12"))
            .expect("link .12");
        std::os::unix::fs::symlink("libcublas.so.12", lib.join("libcublas.so")).expect("link .so");

        merge_component_dir(&component, &cuda_dir).expect("merge component with symlinks");

        let merged_lib = cuda_dir.join("lib");
        // The links are still links (not duplicated), and they still resolve to the real file.
        assert!(
            merged_lib
                .join("libcublas.so")
                .symlink_metadata()
                .expect("stat .so")
                .file_type()
                .is_symlink(),
            "libcublas.so must remain a symlink, not a copied file"
        );
        assert!(
            merged_lib
                .join("libcublas.so.12")
                .symlink_metadata()
                .expect("stat .12")
                .file_type()
                .is_symlink()
        );
        // Following the chain reaches the real file, so linking/loading still works.
        assert_eq!(
            fs::read(merged_lib.join("libcublas.so"))
                .expect("read through the link chain")
                .len(),
            4096
        );
        // And only one real file exists on disk, not three.
        let real_files = fs::read_dir(&merged_lib)
            .expect("read merged lib")
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().expect("type").is_file())
            .count();
        assert_eq!(real_files, 1, "the merge must not duplicate the real .so");
    }
}
