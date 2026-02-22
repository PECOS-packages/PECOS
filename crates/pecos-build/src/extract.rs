//! Archive extraction utilities
//!
//! Provides functions for extracting archives to various locations:
//! - `extract_archive()` - Extract to a specified directory (for legacy/custom use)
//! - `extract_to_deps()` - Extract to `~/.pecos/deps/` (recommended for build scripts)

use crate::errors::{Error, Result};
use crate::home::{get_deps_dir, get_tmp_dir};
use std::fs;
use std::path::{Path, PathBuf};

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
