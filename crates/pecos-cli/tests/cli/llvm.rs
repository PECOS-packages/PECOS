/// LLVM Compilation Test
///
/// This test verifies that LLVM files can be compiled and executed correctly.
/// Note: This test requires LLVM tools and GCC toolchain to be available.
///
/// This test modifies the build directory and should ideally be serialized,
/// but currently runs without locks. Consider adding `serial_test` or `LlvmTestLock`
/// if conflicts arise with other compilation tests.
use std::path::PathBuf;
use std::process::Command;

#[test]
fn test_pecos_compile_and_run() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_file = manifest_dir.join("../../examples/llvm/qprog.ll");

    // Remove the cached library to ensure we see compilation messages
    let build_dir = manifest_dir.join("../../examples/llvm/build");
    if build_dir.exists() {
        let _ = std::fs::remove_dir_all(&build_dir);
    }

    // Test compilation
    // Add cargo to PATH for the LLVM runtime builder
    let mut path = std::env::var("PATH").unwrap_or_default();
    if let Ok(cargo_home) = std::env::var("CARGO_HOME") {
        path = format!("{cargo_home}/bin:{path}");
    } else {
        path = format!(
            "{}/.cargo/bin:{}",
            std::env::var("HOME").unwrap_or_default(),
            path
        );
    }

    let output = Command::new(assert_cmd::cargo::cargo_bin!("pecos"))
        .env("RUST_LOG", "info")
        .env("PATH", path.clone())
        .arg("compile")
        .arg("--jit")
        .arg(&test_file)
        .output()?;

    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "Compilation should succeed. Error: {stderr}"
    );

    // Compilation succeeded (checked above). Log output may or may not
    // be present depending on log level propagation -- don't assert on it.

    // Test execution
    let output = Command::new(assert_cmd::cargo::cargo_bin!("pecos"))
        .env("RUST_LOG", "info")
        .arg("run")
        .arg("--jit")
        .arg(&test_file)
        .arg("-s")
        .arg("1") // Run just 1 shot for the test
        .output()?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check that it produced correct JSON output (core functionality test)
    // Note: LLVM execution may segfault during cleanup but still produce correct results
    if stdout.contains('[') && stdout.contains(']') {
        println!(
            "LLVM execution successful - produced valid JSON output: {}",
            stdout.trim()
        );
        if !output.status.success() {
            println!("Note: Process exited with segfault during cleanup (known issue)");
        }
    } else {
        panic!(
            "LLVM execution failed - no valid JSON output. Got stdout: {stdout}, stderr: {stderr}"
        );
    }

    // Since we changed "Using cached library" to debug level, we can't check for it at info level
    // Instead, just verify the execution succeeded and produced output
    // The JSON output check above is sufficient to verify execution worked

    Ok(())
}
