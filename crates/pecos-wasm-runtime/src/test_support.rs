use std::thread;
use std::time::{Duration, Instant};

// Lifecycle bugs must fail rather than hang the full test process.
pub(crate) fn watchdog(name: &str) -> bool {
    if std::env::var("PECOS_WASM_POOL_TEST").as_deref() == Ok(name) {
        return true;
    }
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([name, "--exact"])
        .env("PECOS_WASM_POOL_TEST", name)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if child.try_wait().unwrap().is_some() {
            let output = child.wait_with_output().unwrap();
            let report = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                output.status.success(),
                "lifecycle child failed: {name}\n{report}"
            );
            // libtest exits successfully when the filter matches nothing, so a
            // stale name would otherwise skip the test body silently.
            assert!(
                report.contains("test result: ok. 1 passed"),
                "lifecycle child did not run exactly one test: {name}\n{report}"
            );
            return false;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("lifecycle child timed out: {name}");
        }
        thread::sleep(Duration::from_millis(10));
    }
}
