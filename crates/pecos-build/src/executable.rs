// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Executable lookup shared by external-tool detectors.

use std::path::{Path, PathBuf};

/// Attempts before [`wait_until_executable`] gives up, so a genuinely unusable
/// file reports failure instead of looping. Backoff is capped, giving a budget
/// of roughly half a second.
const MAX_EXEC_WAIT_ATTEMPTS: u32 = 20;

/// Run `path` with `args` until it is not reported as busy, and return whether
/// it became executable.
///
/// Writing a file and immediately executing it races with process creation
/// elsewhere in the same process. A concurrent `fork` inherits the still-open
/// write descriptor, and `execve` on that file reports `ETXTBSY` until the
/// child completes its own `exec` and the inherited descriptor closes.
/// `O_CLOEXEC` does not avoid this, because the descriptor is still open when
/// the kernel performs the check (rust-lang/rust#39186). The condition cannot
/// be prevented from inside a multi-threaded process, and writing to a
/// temporary name and renaming does not help either, because the check tracks
/// the inode rather than the path.
///
/// One successful execution closes the window permanently for this path. It
/// proves every child that inherited the write descriptor has finished
/// exec'ing, and nothing opens the file for writing again afterwards, so this
/// is a one-time barrier rather than a standing retry.
///
/// A program that runs and exits non-zero still counts as executable.
pub(crate) fn wait_until_executable(path: &Path, args: &[&str]) -> bool {
    use std::io::ErrorKind;
    use std::process::Command;

    for attempt in 0..MAX_EXEC_WAIT_ATTEMPTS {
        match Command::new(path).args(args).output() {
            Ok(_) => return true,
            Err(error) if error.kind() == ErrorKind::ExecutableFileBusy => {
                std::thread::sleep(std::time::Duration::from_millis(1 << attempt.min(5)));
            }
            Err(error) => {
                log::warn!("Could not run {}: {error}", path.display());
                return false;
            }
        }
    }
    log::warn!(
        "{} was still busy after {MAX_EXEC_WAIT_ATTEMPTS} attempts",
        path.display()
    );
    false
}

/// Resolve an executable via `PATH` using the caller's platform suffix policy.
#[must_use]
pub(crate) fn which_in_path(name: &str, extensions: &[&str]) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        for ext in extensions {
            let candidate = dir.join(format!("{name}{ext}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(all(test, unix))]
mod tests {
    use super::wait_until_executable;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Writing a script and executing it immediately fails with `ETXTBSY` at a
    /// double-digit rate when other threads are spawning processes. Reproduce
    /// that load and assert the barrier absorbs it: without the wait inside
    /// [`wait_until_executable`] this fails within a few iterations.
    #[test]
    fn newly_written_scripts_are_executable_while_other_threads_spawn() {
        let dir = std::env::temp_dir().join(format!("pecos_exec_wait_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("Should create probe dir");

        let failures = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..600 {
                        let _ = Command::new("/bin/true").output();
                    }
                });
            }
            for worker in 0..4 {
                let dir = &dir;
                let failures = &failures;
                scope.spawn(move || {
                    for i in 0..150 {
                        let path = dir.join(format!("stub-{worker}-{i}"));
                        fs::write(&path, "#!/bin/sh\necho 21.1.8\n").expect("Should write stub");
                        let mut permissions =
                            fs::metadata(&path).expect("Should stat stub").permissions();
                        permissions.set_mode(0o755);
                        fs::set_permissions(&path, permissions).expect("Should chmod stub");

                        if !wait_until_executable(&path, &["--version"]) {
                            failures.fetch_add(1, Ordering::Relaxed);
                        }
                        let _ = fs::remove_file(&path);
                    }
                });
            }
        });

        let _ = fs::remove_dir_all(&dir);
        assert_eq!(
            failures.load(Ordering::Relaxed),
            0,
            "a freshly written script could not be executed"
        );
    }
}
