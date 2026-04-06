//! Download utilities with caching and integrity verification

use crate::errors::{Error, Result};
use crate::home::get_cache_dir;
use std::fs;

/// Download info with URL and expected SHA256
pub struct DownloadInfo {
    /// Name of the dependency
    pub name: String,
    /// Version string (used for cache naming)
    pub version: String,
    /// URL to download from
    pub url: String,
    /// Expected SHA256 hash
    pub sha256: String,
}

/// Download a file with caching and integrity verification
///
/// Downloads are cached in `~/.pecos/cache/` and verified with SHA256.
///
/// # Errors
///
/// Returns an error if unable to download the file or if verification fails
pub fn download_cached(info: &DownloadInfo) -> Result<Vec<u8>> {
    let cache_dir = get_cache_dir()?;
    // Use version for cache naming (truncate to 12 chars for commits)
    let version_short = &info.version[..12.min(info.version.len())];
    let cache_file = cache_dir.join(format!("{}-{}.tar.gz", info.name, version_short));

    // Check if we have a valid cached file
    if cache_file.exists() {
        match fs::read(&cache_file) {
            Ok(data) => {
                if verify_sha256(&data, &info.sha256).is_ok() {
                    return Ok(data);
                }
                log::warn!("Cached file corrupted, re-downloading");
                let _ = fs::remove_file(&cache_file);
            }
            Err(e) => {
                log::warn!("Failed to read cached file: {e}, re-downloading");
                let _ = fs::remove_file(&cache_file);
            }
        }
    }

    // Download fresh with timeout and retry logic
    log::info!("Downloading {} (will be cached)", info.name);

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| Error::Http(e.to_string()))?;

    // Try download with retries using exponential backoff
    let max_retries = 5;
    let base_delay_secs = 10;
    let mut last_error = String::new();

    for attempt in 1..=max_retries {
        if attempt > 1 {
            let delay_secs = base_delay_secs * (1 << (attempt - 2));
            log::warn!(
                "Retry attempt {}/{} for {} (waiting {}s)",
                attempt,
                max_retries,
                info.name,
                delay_secs
            );
            std::thread::sleep(std::time::Duration::from_secs(delay_secs));
        }

        match client.get(&info.url).send() {
            Ok(response) => {
                let status = response.status();
                if !status.is_success() {
                    last_error = format!("Failed with status: {status}");
                    if status.is_server_error() {
                        log::warn!("Server error ({status}), will retry if attempts remain");
                    }
                    continue;
                }

                match response.bytes() {
                    Ok(bytes) => {
                        let data = bytes.to_vec();

                        if verify_sha256(&data, &info.sha256).is_ok() {
                            fs::write(&cache_file, &data)?;
                            log::info!("Cached to {}", cache_file.display());
                            return Ok(data);
                        }
                        last_error = "SHA256 verification failed".to_string();
                    }
                    Err(e) => {
                        last_error = format!("Failed to read response body: {e}");
                    }
                }
            }
            Err(e) => {
                last_error = format!("Request failed: {e}");
            }
        }
    }

    Err(Error::Download(format!(
        "Failed to download {} after {} attempts: {}",
        info.name, max_retries, last_error
    )))
}

/// Verify SHA256 hash of data
fn verify_sha256(data: &[u8], expected: &str) -> Result<String> {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    Digest::update(&mut hasher, data);
    let result = hasher.finalize();
    let actual = result.iter().fold(String::new(), |mut s, b| {
        use std::fmt::Write;
        write!(s, "{b:02x}").unwrap();
        s
    });

    if actual == expected {
        Ok(actual)
    } else {
        Err(Error::Sha256Mismatch {
            expected: expected.to_string(),
            actual,
        })
    }
}

/// Download multiple files concurrently
///
/// # Errors
///
/// Returns an error if any download fails
///
/// # Errors
///
/// Returns an error if any download fails or a thread panics.
///
/// # Panics
///
/// Panics if an internal mutex is poisoned (indicates a prior thread panic).
pub fn download_all_cached(downloads: Vec<DownloadInfo>) -> Result<Vec<(String, Vec<u8>)>> {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let results = Arc::new(Mutex::new(Vec::new()));
    let errors = Arc::new(Mutex::new(Vec::new()));

    let handles: Vec<_> = downloads
        .into_iter()
        .map(|info| {
            let results = Arc::clone(&results);
            let errors = Arc::clone(&errors);

            thread::spawn(move || match download_cached(&info) {
                Ok(data) => {
                    results
                        .lock()
                        .expect("results mutex poisoned")
                        .push((info.name.clone(), data));
                }
                Err(e) => {
                    errors
                        .lock()
                        .expect("errors mutex poisoned")
                        .push(format!("{}: {}", info.name, e));
                }
            })
        })
        .collect();

    for handle in handles {
        handle
            .join()
            .map_err(|_| Error::Download("download thread panicked".to_string()))?;
    }

    let errors = errors.lock().expect("errors mutex poisoned");
    if !errors.is_empty() {
        return Err(Error::Download(format!(
            "Download failures:\n{}",
            errors.join("\n")
        )));
    }
    drop(errors);

    let results = Arc::try_unwrap(results)
        .map_err(|_| Error::Download("unexpected outstanding Arc reference".to_string()))?;
    Ok(results.into_inner().expect("results mutex poisoned"))
}
