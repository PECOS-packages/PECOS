//! Ownership, publication staging, and advisory locks for the program cache.

use super::InterfaceError;
use log::{debug, warn};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime};

pub(super) const LIB_SUFFIX: &str = if cfg!(target_os = "windows") {
    "dll"
} else {
    "so"
};
const TOKEN_LEN: usize = 6;
static CLEANUP_DONE: OnceLock<()> = OnceLock::new();
static UNSUPPORTED_WARNING: OnceLock<()> = OnceLock::new();

// Windows winerror.h: ERROR_NOT_SUPPORTED=50 and ERROR_INVALID_FUNCTION=1.
#[cfg(windows)]
pub(super) const ERROR_NOT_SUPPORTED: i32 = 50;
#[cfg(windows)]
pub(super) const ERROR_INVALID_FUNCTION: i32 = 1;

fn locking_unsupported(error: &std::io::Error) -> bool {
    if error.kind() == std::io::ErrorKind::Unsupported {
        return true;
    }
    // ENOLCK: no lock manager (e.g. NFS without lockd). ENOTSUP: macOS keeps it distinct
    // from EOPNOTSUPP, which std already reports as Unsupported.
    #[cfg(unix)]
    if matches!(error.raw_os_error(), Some(libc::ENOLCK | libc::ENOTSUP)) {
        return true;
    }
    #[cfg(windows)]
    if matches!(
        error.raw_os_error(),
        Some(ERROR_NOT_SUPPORTED | ERROR_INVALID_FUNCTION)
    ) {
        return true;
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EntryKind {
    Library,
    Manifest,
    Lock,
    Staging,
}

/// Recognize only names this cache owns, including tempfile's alphanumeric token.
pub(super) fn parse_entry(name: &str) -> Option<(&str, EntryKind)> {
    let (body, staging) = name.strip_prefix(".program_").map_or_else(
        || Some((name.strip_prefix("program_")?, false)),
        |s| Some((s, true)),
    )?;
    let (digest, tail) = body.split_once('.')?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return None;
    }
    let kind = if staging {
        let token = tail.strip_suffix(".tmp")?;
        if token.len() != TOKEN_LEN || !token.bytes().all(|b| b.is_ascii_alphanumeric()) {
            return None;
        }
        EntryKind::Staging
    } else {
        match tail {
            s if s == LIB_SUFFIX => EntryKind::Library,
            "manifest" => EntryKind::Manifest,
            "lock" => EntryKind::Lock,
            _ => return None,
        }
    };
    Some((digest, kind))
}

pub(super) fn get_persistent_cache_dir() -> Result<PathBuf, InterfaceError> {
    let root = std::env::var_os("PECOS_CACHE_DIR").map_or_else(
        || std::env::temp_dir().join("pecos_compiled_cache"),
        PathBuf::from,
    );
    let cache_dir = root.join("qis-programs");
    std::fs::create_dir_all(&cache_dir)
        .map_err(|e| InterfaceError::LoadError(format!("Failed to create cache directory: {e}")))?;
    // Cleanup never runs while this process holds a compilation lock.
    CLEANUP_DONE.get_or_init(|| cleanup_old_cache_files(&cache_dir, 24 * 60 * 60));
    Ok(cache_dir)
}

pub(super) fn staging_dir(cache_dir: &Path, digest: &str) -> std::io::Result<tempfile::TempDir> {
    tempfile::Builder::new()
        .prefix(&format!(".program_{digest}."))
        .rand_bytes(TOKEN_LEN)
        .suffix(".tmp")
        .tempdir_in(cache_dir)
}

/// Closing the independently opened handle releases the OS lock, even on exit.
pub(super) struct CompilationLock {
    _file: File,
}

pub(super) enum LockOutcome {
    Acquired(CompilationLock),
    Unsupported,
    TimedOut,
}

impl CompilationLock {
    fn open(cache_dir: &Path, digest: &str) -> std::io::Result<File> {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(cache_dir.join(format!("program_{digest}.lock")))
    }

    fn try_lock(file: &File) -> Result<(), TryLockError> {
        #[cfg(test)]
        super::cache_faults::lock_error()?;
        file.try_lock()
    }

    pub(super) fn acquire(
        cache_dir: &Path,
        digest: &str,
        budget: Duration,
    ) -> std::io::Result<LockOutcome> {
        let file = Self::open(cache_dir, digest)?;
        let start = Instant::now();
        loop {
            match Self::try_lock(&file) {
                Ok(()) => return Ok(LockOutcome::Acquired(Self { _file: file })),
                Err(TryLockError::WouldBlock) => {
                    #[cfg(test)]
                    super::cache_faults::point(super::cache_faults::Site::LockPoll, cache_dir);
                    if start.elapsed() >= budget {
                        return Ok(LockOutcome::TimedOut);
                    }
                    std::thread::sleep(
                        Duration::from_millis(10).min(budget.saturating_sub(start.elapsed())),
                    );
                }
                Err(TryLockError::Error(e)) if locking_unsupported(&e) => {
                    UNSUPPORTED_WARNING.get_or_init(|| {
                        warn!("Program cache filesystem does not support locking; compiling without a lock");
                    });
                    return Ok(LockOutcome::Unsupported);
                }
                Err(TryLockError::Error(e)) => return Err(e),
            }
        }
    }
}

fn is_old(path: &Path, max_age_secs: u64) -> bool {
    std::fs::symlink_metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .is_some_and(|age| age.as_secs() > max_age_secs)
}

/// Best effort, one nonblocking lock attempt per digest. Lock files are permanent.
pub(super) fn cleanup_old_cache_files(cache_dir: &Path, max_age_secs: u64) {
    let Ok(entries) = std::fs::read_dir(cache_dir) else {
        return;
    };
    let mut candidates = BTreeMap::<String, Vec<(PathBuf, EntryKind)>>::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some((digest, kind)) = name.to_str().and_then(parse_entry) else {
            continue;
        };
        if kind != EntryKind::Lock && is_old(&entry.path(), max_age_secs) {
            candidates
                .entry(digest.to_owned())
                .or_default()
                .push((entry.path(), kind));
        }
    }
    for (digest, paths) in candidates {
        #[cfg(test)]
        super::cache_faults::point(super::cache_faults::Site::CleanupObserved, cache_dir);
        let file = match CompilationLock::open(cache_dir, &digest) {
            Ok(file) => file,
            Err(e) => {
                warn!("Cannot open cleanup lock for {digest}: {e}");
                continue;
            }
        };
        match CompilationLock::try_lock(&file) {
            Ok(()) => (),
            Err(TryLockError::WouldBlock) => continue,
            Err(TryLockError::Error(e)) if locking_unsupported(&e) => {
                debug!("Locking unsupported for cleanup of {digest}: {e}");
                continue;
            }
            Err(TryLockError::Error(e)) => {
                warn!("Cannot lock program cache for cleanup of {digest}: {e}");
                continue;
            }
        }
        let _lock = CompilationLock { _file: file };
        #[cfg(test)]
        super::cache_faults::point(super::cache_faults::Site::CleanupLocked, cache_dir);
        for (path, kind) in paths {
            // A publisher may have replaced an old entry before we acquired its lock.
            if !is_old(&path, max_age_secs) {
                continue;
            }
            debug!("Removing old program cache entry: {}", path.display());
            let result = if kind == EntryKind::Staging {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            };
            if let Err(e) = result {
                debug!("Cannot remove program cache entry {}: {e}", path.display());
            }
        }
    }
}
