//! Archive extraction utilities
//!
//! Provides functions for extracting archives to various locations:
//! - `extract_archive()` - Extract to a specified directory (for legacy/custom use)
//! - `extract_to_deps()` - Extract to `~/.pecos/deps/` (recommended for build scripts)
//! - `contained_entry_path()` - Resolve an archive entry name inside a destination

use crate::errors::{Error, Result};
use crate::home::{get_deps_dir, get_tmp_dir};
use std::fs;
use std::path::{Component, Path, PathBuf};

/// Names Windows resolves as devices in every directory, so `NUL.txt` is the null
/// device rather than a file. Matched case-insensitively against the part of a
/// component before its first `.`.
const WINDOWS_DEVICE_NAMES: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// Resolves an archive entry name to a path inside `dest`, rejecting escapes.
///
/// Archive entry names are attacker-controlled data: a crafted archive can name an
/// entry `../../etc/foo` or `/etc/foo` and a naive `dest.join(name)` will then write
/// outside the extraction directory. Callers must route every entry name through this
/// function before creating anything on disk.
///
/// The `tar` crate validates this inside `unpack`, and the `zip` crate exposes
/// `enclosed_name` for it, so only archive readers without such a guard need this.
///
/// Containment here is **lexical**. It assumes nothing under `dest` is a symlink or
/// junction: this function cannot stop a write that travels through a link an earlier
/// process placed inside the extraction directory.
///
/// # Errors
///
/// Returns [`Error::Archive`] if the entry name is absolute, contains a `..` component,
/// contains a Windows prefix such as `C:`, has a component with trailing dots or spaces,
/// or names a Windows device.
pub fn contained_entry_path(dest: &Path, entry_name: &str) -> Result<PathBuf> {
    // Treat both separators as separating on every platform: a Windows-built archive
    // read on Unix would otherwise carry "..\\.." through as a single opaque name.
    let normalized = entry_name.replace('\\', "/");
    let candidate = Path::new(&normalized);

    for component in candidate.components() {
        match component {
            Component::Normal(name) => {
                let name = name.to_string_lossy();

                // Windows strips trailing dots and spaces while resolving a path, so a
                // component like ".. " arrives here as an innocuous Normal and is only
                // turned back into ".." by the filesystem -- after this check would have
                // passed. Rejecting any trailing dot or space closes that class without
                // having to model Win32 normalization; such names are unrepresentable on
                // Windows anyway and do not occur in the archives this reads.
                if name.ends_with('.') || name.ends_with(' ') {
                    return Err(Error::Archive(format!(
                        "archive entry has a component with trailing dots or spaces, \
                         which some filesystems resolve to a different path: {entry_name}"
                    )));
                }

                let stem = name.split('.').next().unwrap_or(&name);
                if WINDOWS_DEVICE_NAMES
                    .iter()
                    .any(|device| stem.eq_ignore_ascii_case(device))
                {
                    return Err(Error::Archive(format!(
                        "archive entry names a reserved device: {entry_name}"
                    )));
                }
            }
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(Error::Archive(format!(
                    "archive entry escapes the extraction directory: {entry_name}"
                )));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(Error::Archive(format!(
                    "archive entry is an absolute path: {entry_name}"
                )));
            }
        }
    }

    if candidate.as_os_str().is_empty() {
        return Err(Error::Archive("archive entry has an empty name".into()));
    }

    Ok(dest.join(candidate))
}

/// Extract a tar.gz or tar.bz2 archive
///
/// Automatically detects archive format by magic bytes and extracts to the specified directory.
///
/// # Arguments
///
/// * `data` - The archive data bytes
/// * `out_dir` - Directory to extract into
/// * `expected_dir_name` - Optional name for the extracted directory (defaults to "extracted")
///
/// # Errors
///
/// Returns an error if extraction fails or the expected directory is not found
pub fn extract_archive(
    data: &[u8],
    out_dir: &Path,
    expected_dir_name: Option<&str>,
) -> Result<PathBuf> {
    use tar::Archive;

    // Detect archive format by magic bytes
    let mut archive = if data.len() >= 3 && data[0] == 0x1f && data[1] == 0x8b && data[2] == 0x08 {
        // gzip magic bytes
        use flate2::read::GzDecoder;
        let tar = GzDecoder::new(data);
        Archive::new(Box::new(tar) as Box<dyn std::io::Read>)
    } else if data.len() >= 3 && &data[0..3] == b"BZh" {
        // bzip2 magic bytes
        use bzip2::read::BzDecoder;
        let tar = BzDecoder::new(data);
        Archive::new(Box::new(tar) as Box<dyn std::io::Read>)
    } else {
        return Err(Error::Archive(
            "Unknown archive format - not gzip or bzip2".to_string(),
        ));
    };

    // Extract to temporary directory first under ~/.pecos/tmp/
    // This keeps all PECOS files in one place and makes cleanup easier
    let pecos_tmp = get_tmp_dir()?;
    let temp_dir = pecos_tmp.join(format!("extract_{}", std::process::id()));
    fs::create_dir_all(&temp_dir)?;

    // Configure archive for Windows compatibility
    archive.set_preserve_permissions(false);
    archive.set_unpack_xattrs(false);
    archive.unpack(&temp_dir)?;

    // Find the extracted directory
    let entries = fs::read_dir(&temp_dir)?;
    let extracted_dir = entries
        .filter_map(std::result::Result::ok)
        .find(|e| e.file_type().ok().is_some_and(|t| t.is_dir()))
        .ok_or_else(|| Error::Archive("No directory found in archive".to_string()))?
        .path();

    // Move to final location
    let final_name = expected_dir_name.unwrap_or("extracted");
    let final_dir = out_dir.join(final_name);

    // Ensure parent directory exists
    fs::create_dir_all(out_dir)?;

    if final_dir.exists() {
        fs::remove_dir_all(&final_dir)?;
    }

    // On Windows, use copy instead of rename to avoid path length issues
    #[cfg(windows)]
    {
        copy_dir_all(&extracted_dir, &final_dir)?;
        // Temp dir cleanup can fail on Windows due to antivirus locks or
        // concurrent access - this is non-fatal since the extraction succeeded.
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[cfg(not(windows))]
    {
        fs::rename(extracted_dir, &final_dir)?;
        fs::remove_dir_all(&temp_dir)?;
    }

    Ok(final_dir)
}

/// Recursively copy a directory and all its contents
#[cfg(windows)]
fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if entry.file_type()?.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

/// Extract an archive to `~/.pecos/deps/<dir_name>/`
///
/// This is the recommended extraction function for build scripts.
/// Archives are extracted to a persistent location that survives `cargo clean`.
///
/// # Arguments
///
/// * `data` - The archive data bytes
/// * `dir_name` - Name for the extracted directory (e.g., "qulacs-abc123")
///
/// # Returns
///
/// The path to the extracted directory (`~/.pecos/deps/<dir_name>/`)
///
/// # Errors
///
/// Returns an error if extraction fails
pub fn extract_to_deps(data: &[u8], dir_name: &str) -> Result<PathBuf> {
    let deps_dir = get_deps_dir()?;
    extract_archive(data, &deps_dir, Some(dir_name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_entry_names_resolve_inside_the_destination() {
        let path = contained_entry_path(Path::new("/out"), "bin/nvcc").unwrap();
        assert_eq!(path, Path::new("/out/bin/nvcc"));
    }

    #[test]
    fn windows_separators_resolve_component_wise() {
        let path = contained_entry_path(Path::new("/out"), "bin\\nvcc.exe").unwrap();
        assert_eq!(path, Path::new("/out/bin/nvcc.exe"));
    }

    #[test]
    fn parent_components_are_rejected() {
        for name in [
            "../escape",
            "bin/../../escape",
            "..\\escape",
            "bin\\..\\..\\escape",
        ] {
            let error = contained_entry_path(Path::new("/out"), name).unwrap_err();
            assert!(
                matches!(error, Error::Archive(message) if message.contains("escapes")),
                "expected an escape rejection for {name}"
            );
        }
    }

    #[test]
    fn absolute_entry_names_are_rejected() {
        for name in ["/etc/passwd", "\\windows\\system32\\evil.dll"] {
            let error = contained_entry_path(Path::new("/out"), name).unwrap_err();
            assert!(
                matches!(error, Error::Archive(message) if message.contains("absolute")),
                "expected an absolute-path rejection for {name}"
            );
        }
    }

    #[test]
    fn trailing_space_or_dot_components_are_rejected() {
        // Windows strips trailing dots and spaces during path resolution, so ".. " would
        // become ".." after this check and escape the destination.
        for name in [
            ".. /nvcc",
            "nvcc/.. /x",
            "...",
            "nvcc/... /x",
            "trailing.",
            "trailing ",
        ] {
            let error = contained_entry_path(Path::new("/out"), name).unwrap_err();
            assert!(
                matches!(error, Error::Archive(message) if message.contains("trailing dots or spaces")),
                "expected a trailing dot/space rejection for {name:?}"
            );
        }
    }

    #[test]
    fn windows_device_names_are_rejected() {
        for name in [
            "nvcc/NUL.txt",
            "nvcc/COM1",
            "con",
            "AUX",
            "bin/lpt9.dll",
            "NuL",
        ] {
            let error = contained_entry_path(Path::new("/out"), name).unwrap_err();
            assert!(
                matches!(error, Error::Archive(message) if message.contains("reserved device")),
                "expected a device-name rejection for {name:?}"
            );
        }
    }

    #[test]
    fn ordinary_names_containing_device_words_are_allowed() {
        // Only an exact device stem is reserved; these merely contain the letters.
        for name in ["console.dll", "nulled.txt", "bin/comment.h", "auxiliary/x"] {
            assert!(
                contained_entry_path(Path::new("/out"), name).is_ok(),
                "{name} should be allowed"
            );
        }
    }

    #[test]
    fn empty_entry_names_are_rejected() {
        assert!(contained_entry_path(Path::new("/out"), "").is_err());
    }

    #[test]
    fn current_directory_components_are_allowed() {
        let path = contained_entry_path(Path::new("/out"), "./bin/nvcc").unwrap();
        assert_eq!(path, Path::new("/out/bin/nvcc"));
    }
}
