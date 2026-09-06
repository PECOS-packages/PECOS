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
use std::process::{Command, Output};

/// Attempts before [`run_when_executable`] gives up, so a genuinely unusable
/// file reports failure instead of looping. Backoff is capped, giving a budget
/// of roughly half a second.
const MAX_EXEC_WAIT_ATTEMPTS: u32 = 20;

/// Run `path` with `args` until it is no longer reported as busy, and return
/// the output of the run that succeeded.
///
/// Writing a file and immediately executing it races with process creation
/// elsewhere in the same process. A concurrent `fork` inherits the still-open
/// write descriptor, and `execve` on that file reports `ETXTBSY` until the last
/// reference to that writable file description is released. `O_CLOEXEC` does
/// not avoid this, because the descriptor is still open when the kernel
/// performs the check (rust-lang/rust#114554). Writing to a temporary name and
/// renaming does not help either, because the check tracks the inode rather
/// than the path.
///
/// This is a barrier against writers that already exist, not a permanent
/// guarantee about the path. A successful execution establishes that no
/// writable description of that inode remains open, so writers inherited from
/// earlier forks cannot recreate one. It does not promise the file stays
/// executable: a later writer, including one reaching the same inode through a
/// hard link, reopens the window, and replacing the inode at that path is a
/// different file entirely. Both callers write a stub once and never rewrite
/// it, which is the situation this is for.
///
/// A program that runs and exits non-zero still counts as executable.
///
/// # Errors
/// Returns the operating system's error when the file cannot be run for a
/// reason other than being busy, or an [`ErrorKind::ExecutableFileBusy`] error
/// naming the attempt count when it stayed busy for the whole budget. Callers
/// report the cause; a bare boolean would leave it only in the log, which a
/// library consumer need not have initialized.
pub(crate) fn run_when_executable(path: &Path, args: &[&str]) -> std::io::Result<Output> {
    retry_while_busy(path, &mut RealAttempts, |runner| runner.run(path, args))
}

/// The two effects the retry loop performs, so tests can supply them directly.
///
/// Timing-based tests of this loop were repeatedly vacuous: a fixed delay long
/// enough to outlast the test's own writer passes without ever retrying. Making
/// the attempt and the sleep injectable lets the retry behaviour be asserted
/// exactly, on every platform, without racing the scheduler.
trait Attempts {
    fn run(&mut self, path: &Path, args: &[&str]) -> std::io::Result<Output>;
    fn sleep(&mut self, duration: std::time::Duration);
}

struct RealAttempts;

impl Attempts for RealAttempts {
    fn run(&mut self, path: &Path, args: &[&str]) -> std::io::Result<Output> {
        Command::new(path).args(args).output()
    }
    fn sleep(&mut self, duration: std::time::Duration) {
        std::thread::sleep(duration);
    }
}

fn retry_while_busy<A: Attempts>(
    path: &Path,
    attempts: &mut A,
    mut run: impl FnMut(&mut A) -> std::io::Result<Output>,
) -> std::io::Result<Output> {
    use std::io::{Error, ErrorKind};

    for attempt in 0..MAX_EXEC_WAIT_ATTEMPTS {
        match run(attempts) {
            Ok(output) => return Ok(output),
            Err(error) if error.kind() == ErrorKind::ExecutableFileBusy => {
                // No point sleeping when no attempt remains.
                if attempt + 1 == MAX_EXEC_WAIT_ATTEMPTS {
                    break;
                }
                attempts.sleep(std::time::Duration::from_millis(1 << attempt.min(5)));
            }
            Err(error) => {
                log::warn!("Could not run {}: {error}", path.display());
                return Err(error);
            }
        }
    }
    log::warn!(
        "{} was still busy after {MAX_EXEC_WAIT_ATTEMPTS} attempts",
        path.display()
    );
    Err(Error::new(
        ErrorKind::ExecutableFileBusy,
        format!(
            "{} was still busy after {MAX_EXEC_WAIT_ATTEMPTS} attempts",
            path.display()
        ),
    ))
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
    use super::run_when_executable;
    use std::fs;
    use std::io::ErrorKind;
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{Attempts, MAX_EXEC_WAIT_ATTEMPTS, retry_while_busy};
    use std::io::Error;
    use std::os::unix::process::ExitStatusExt;
    use std::process::{ExitStatus, Output};
    use std::time::Duration;

    /// Reports busy for the first `busy_before_success` attempts, then hands
    /// back `result`. Records every attempt and every requested sleep.
    struct ScriptedAttempts {
        busy_before_success: u32,
        result: Option<std::io::Result<Output>>,
        runs: u32,
        sleeps: Vec<Duration>,
    }

    impl ScriptedAttempts {
        fn busy_then(busy_before_success: u32, result: std::io::Result<Output>) -> Self {
            Self {
                busy_before_success,
                result: Some(result),
                runs: 0,
                sleeps: Vec::new(),
            }
        }
        fn always_busy() -> Self {
            Self {
                busy_before_success: u32::MAX,
                result: None,
                runs: 0,
                sleeps: Vec::new(),
            }
        }
    }

    impl Attempts for ScriptedAttempts {
        fn run(&mut self, _path: &std::path::Path, _args: &[&str]) -> std::io::Result<Output> {
            self.runs += 1;
            if self.runs <= self.busy_before_success {
                return Err(Error::from_raw_os_error(26));
            }
            self.result
                .take()
                .expect("the scripted result is consumed once")
        }
        fn sleep(&mut self, duration: Duration) {
            self.sleeps.push(duration);
        }
    }

    fn output(stdout: &str, stderr: &str, code: i32) -> Output {
        Output {
            status: ExitStatus::from_raw(code << 8),
            stdout: stdout.as_bytes().to_vec(),
            stderr: stderr.as_bytes().to_vec(),
        }
    }

    /// The loop must keep re-attempting while busy, and hand back the
    /// successful attempt's result untouched: same stdout, stderr and exit
    /// status, from exactly one successful execution.
    #[test]
    fn a_busy_target_is_retried_and_its_eventual_output_returned_verbatim() {
        let mut attempts = ScriptedAttempts::busy_then(3, Ok(output("21.1.8", "a warning", 3)));
        let result = retry_while_busy(std::path::Path::new("/probe"), &mut attempts, |a| {
            a.run(std::path::Path::new("/probe"), &["--version"])
        })
        .expect("the fourth attempt succeeds");

        assert_eq!(attempts.runs, 4, "three busy attempts then one success");
        assert_eq!(String::from_utf8_lossy(&result.stdout), "21.1.8");
        assert_eq!(String::from_utf8_lossy(&result.stderr), "a warning");
        assert_eq!(
            result.status.code(),
            Some(3),
            "a non-zero exit is preserved"
        );
    }

    /// Backoff doubles and then caps, so a widened budget cannot slip in.
    #[test]
    fn the_backoff_doubles_up_to_the_cap() {
        let mut attempts = ScriptedAttempts::busy_then(5, Ok(output("ok", "", 0)));
        retry_while_busy(std::path::Path::new("/probe"), &mut attempts, |a| {
            a.run(std::path::Path::new("/probe"), &[])
        })
        .expect("succeeds after five busy attempts");
        assert_eq!(
            attempts.sleeps,
            [1, 2, 4, 8, 16].map(Duration::from_millis),
            "backoff should double from 1ms"
        );
    }

    /// A target that never stops being busy is given up on after the bounded
    /// number of attempts, with no sleep after the final one.
    #[test]
    fn a_permanently_busy_target_is_abandoned_after_the_bounded_attempts() {
        let mut attempts = ScriptedAttempts::always_busy();
        let error = retry_while_busy(std::path::Path::new("/probe"), &mut attempts, |a| {
            a.run(std::path::Path::new("/probe"), &[])
        })
        .expect_err("a permanently busy target is not executable");

        assert_eq!(error.kind(), ErrorKind::ExecutableFileBusy);
        assert_eq!(attempts.runs, MAX_EXEC_WAIT_ATTEMPTS);
        assert_eq!(
            attempts.sleeps.len(),
            usize::try_from(MAX_EXEC_WAIT_ATTEMPTS - 1).expect("attempt count fits a usize"),
            "no sleep after the last attempt"
        );
        assert_eq!(
            attempts.sleeps.iter().max(),
            Some(&Duration::from_millis(32)),
            "backoff is capped"
        );
    }

    /// An error that is not "busy" is returned as it came from the operating
    /// system, not reclassified or replaced.
    #[test]
    fn a_non_busy_error_is_returned_unchanged() {
        let mut attempts = ScriptedAttempts::busy_then(0, Err(Error::from_raw_os_error(13)));
        let error = retry_while_busy(std::path::Path::new("/probe"), &mut attempts, |a| {
            a.run(std::path::Path::new("/probe"), &[])
        })
        .expect_err("a permission error is not retried");

        assert_eq!(error.raw_os_error(), Some(13), "the OS error is preserved");
        assert_eq!(attempts.runs, 1, "a non-busy error is not retried");
        assert!(attempts.sleeps.is_empty());
    }

    /// Produce an executable at `name`: a shell stub, or a copy of `source`
    /// when a native binary is wanted.
    #[cfg(target_os = "linux")]
    fn materialise(dir: &std::path::Path, name: &str, source: Option<&str>) -> std::path::PathBuf {
        let Some(binary) = source else {
            return write_stub(dir, name);
        };
        let path = dir.join(name);
        fs::copy(binary, &path).expect("Should copy a native binary");
        let mut permissions = fs::metadata(&path).expect("Should stat copy").permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).expect("Should chmod copy");
        path
    }

    /// Write an executable shell stub and return its path.
    fn write_stub(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        fs::write(&path, "#!/bin/sh\necho 21.1.8 \"$@\"\n").expect("Should write stub");
        let mut permissions = fs::metadata(&path).expect("Should stat stub").permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).expect("Should chmod stub");
        path
    }

    #[test]
    fn a_missing_target_reports_the_operating_system_error() {
        let temp = tempfile::tempdir().expect("Should create probe dir");
        let error = run_when_executable(&temp.path().join("absent"), &["--version"])
            .expect_err("a missing file is not executable");
        assert_eq!(error.kind(), ErrorKind::NotFound);
    }

    #[test]
    fn a_non_executable_target_reports_the_operating_system_error() {
        let temp = tempfile::tempdir().expect("Should create probe dir");
        let path = temp.path().join("not-executable");
        fs::write(&path, "#!/bin/sh\n").expect("Should write file");
        let error = run_when_executable(&path, &["--version"])
            .expect_err("a file without the execute bit is not executable");
        assert_eq!(error.kind(), ErrorKind::PermissionDenied);
    }

    #[test]
    fn the_successful_run_output_is_returned() {
        let temp = tempfile::tempdir().expect("Should create probe dir");
        let path = write_stub(temp.path(), "reports-version");
        let output = run_when_executable(&path, &["--version"]).expect("stub should run");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            "21.1.8 --version"
        );
    }

    /// The caller's arguments are forwarded, not a fixed set. Every current
    /// caller happens to pass `--version`, so without this the contract would
    /// be untested.
    #[test]
    fn the_supplied_arguments_are_forwarded() {
        let temp = tempfile::tempdir().expect("Should create probe dir");
        let path = write_stub(temp.path(), "echoes-args");
        let output =
            run_when_executable(&path, &["--prefix", "--libdir"]).expect("stub should run");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            "21.1.8 --prefix --libdir"
        );
    }

    /// An open writable descriptor is exactly the condition that produces
    /// `ETXTBSY`, so holding one reproduces it with no race at all.
    ///
    /// Linux only. Apple's XNU has the equivalent open-writer check compiled
    /// out, so a held descriptor does not block execution there and this would
    /// assert on a condition the platform never produces.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_held_writable_descriptor_is_reported_rather_than_waited_out() {
        let temp = tempfile::tempdir().expect("Should create probe dir");
        for (name, source) in [("held-script", None), ("held-binary", Some("/bin/true"))] {
            let path = materialise(temp.path(), name, source);

            let writer = fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .expect("Should hold the target open for writing");

            let error = run_when_executable(&path, &["--version"])
                .expect_err("a target held open for writing is not executable");
            assert_eq!(error.kind(), ErrorKind::ExecutableFileBusy, "{name}");

            drop(writer);
            run_when_executable(&path, &["--version"]).unwrap_or_else(|error| {
                panic!("{name} should run once the writer is gone: {error}")
            });
        }
    }

    /// The barrier must keep re-checking until the writer goes away, rather
    /// than letting wall-clock time pass and hoping.
    ///
    /// The writer is released well after any plausible fixed delay, and the
    /// verification happens INSIDE the scope: leaving the scope would join the
    /// releasing thread and establish readiness by itself, which would let a
    /// helper that does nothing pass.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_barrier_waits_for_the_writer_to_be_released() {
        let temp = tempfile::tempdir().expect("Should create probe dir");
        // A shell script and a native binary: recovery from a real busy launch
        // must work for both, so an ELF special case cannot hide here.
        for (name, source) in [
            ("released-script", None),
            ("released-binary", Some("/bin/true")),
        ] {
            let path = materialise(temp.path(), name, source);

            let writer = fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .expect("Should hold the target open for writing");

            std::thread::scope(|scope| {
                scope.spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(250));
                    drop(writer);
                });
                run_when_executable(&path, &["--version"])
                    .unwrap_or_else(|error| panic!("{name} should wait for release: {error}"));
                Command::new(&path)
                    .arg("--version")
                    .output()
                    .unwrap_or_else(|error| panic!("{name} should run once ready: {error}"));
            });
        }
    }

    /// Writing a script and executing it immediately fails with `ETXTBSY` at a
    /// double-digit rate when other threads are spawning processes. Reproduce
    /// that load and assert the barrier absorbs it.
    ///
    /// Each stub is run again after the barrier reports success, because
    /// asserting only on the helper's return value would pass even if the
    /// helper did nothing at all.
    #[test]
    fn newly_written_scripts_are_executable_while_other_threads_spawn() {
        let temp = tempfile::tempdir().expect("Should create probe dir");
        let dir = temp.path().to_path_buf();

        let failures = AtomicUsize::new(0);
        let post_barrier_busy = AtomicUsize::new(0);
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
                let post_barrier_busy = &post_barrier_busy;
                scope.spawn(move || {
                    for i in 0..150 {
                        let path = write_stub(dir, &format!("stub-{worker}-{i}"));
                        if run_when_executable(&path, &["--version"]).is_err() {
                            failures.fetch_add(1, Ordering::Relaxed);
                            let _ = fs::remove_file(&path);
                            continue;
                        }
                        if let Err(error) = Command::new(&path).arg("--version").output() {
                            assert_eq!(
                                error.kind(),
                                ErrorKind::ExecutableFileBusy,
                                "unexpected failure running a stub: {error}"
                            );
                            post_barrier_busy.fetch_add(1, Ordering::Relaxed);
                        }
                        let _ = fs::remove_file(&path);
                    }
                });
            }
        });

        assert_eq!(
            failures.load(Ordering::Relaxed),
            0,
            "a freshly written script could not be executed"
        );
        assert_eq!(
            post_barrier_busy.load(Ordering::Relaxed),
            0,
            "a script was still ETXTBSY after the barrier reported it ready"
        );
    }
}
