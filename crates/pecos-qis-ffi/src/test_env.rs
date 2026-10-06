//! Process isolation for tests of the FFI runtime and its thread-local state.

/// Follow the QIS crate's child-test pattern without linking a second FFI runtime.
pub(crate) fn run_test_in_child(test: &str) -> bool {
    const CHILD_TEST: &str = "PECOS_FFI_CHILD_TEST";
    if std::env::var(CHILD_TEST).as_deref() == Ok(test) {
        return true;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env(CHILD_TEST, test)
        .output()
        .expect("start isolated FFI test");
    assert!(
        output.status.success(),
        "isolated test {test} failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    false
}
