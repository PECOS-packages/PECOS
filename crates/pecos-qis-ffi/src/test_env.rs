//! Process isolation for tests of the FFI runtime and its thread-local state.

struct TestChild {
    process: std::process::Child,
    stdout: Option<std::thread::JoinHandle<std::io::Result<Vec<u8>>>>,
    stderr: Option<std::thread::JoinHandle<std::io::Result<Vec<u8>>>>,
}

impl TestChild {
    fn kill_and_wait(&mut self) -> std::process::ExitStatus {
        self.process.kill().expect("kill child");
        self.process.wait().expect("observe child exit")
    }
}

impl Drop for TestChild {
    fn drop(&mut self) {
        // Also reap the child if the parent's wait or output collection panics.
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

fn drain_child_pipe(
    mut pipe: impl std::io::Read + Send + 'static,
) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>> {
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        pipe.read_to_end(&mut bytes)?;
        Ok(bytes)
    })
}

fn finish_test_child(
    mut child: TestChild,
    budget: std::time::Duration,
) -> Result<std::process::Output, String> {
    let watchdog = std::time::Instant::now();
    let (status, timed_out) = loop {
        if let Some(status) = child.process.try_wait().expect("child status") {
            break (status, false);
        }
        if watchdog.elapsed() >= budget {
            break (child.kill_and_wait(), true);
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    let stdout = child
        .stdout
        .take()
        .expect("stdout reader")
        .join()
        .expect("stdout thread")
        .expect("read stdout");
    let stderr = child
        .stderr
        .take()
        .expect("stderr reader")
        .join()
        .expect("stderr thread")
        .expect("read stderr");
    let output = std::process::Output {
        status,
        stdout,
        stderr,
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if timed_out {
        return Err(format!(
            "child watchdog expired ({status}): {stdout}\n{stderr}"
        ));
    }
    if !status.success() {
        return Err(format!("child failed ({status}): {stdout}\n{stderr}"));
    }
    if !stdout.contains("test result: ok. 1 passed") {
        return Err(format!(
            "child did not run exactly one test ({status}): {stdout}\n{stderr}"
        ));
    }
    Ok(output)
}

/// Follow the QIS crate's child-test pattern without linking a second FFI runtime.
pub(crate) fn run_test_in_child(test: &str) -> bool {
    const CHILD_TEST: &str = "PECOS_FFI_CHILD_TEST";
    if std::env::var(CHILD_TEST).as_deref() == Ok(test) {
        return true;
    }
    let mut process = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env(CHILD_TEST, test)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("start isolated FFI test");
    let stdout = drain_child_pipe(process.stdout.take().expect("stdout pipe"));
    let stderr = drain_child_pipe(process.stderr.take().expect("stderr pipe"));
    let child = TestChild {
        process,
        stdout: Some(stdout),
        stderr: Some(stderr),
    };
    finish_test_child(child, std::time::Duration::from_secs(60))
        .unwrap_or_else(|error| panic!("isolated test {test}: {error}"));
    false
}
