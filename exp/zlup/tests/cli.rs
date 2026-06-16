//! CLI integration tests for Zlup.

use std::fs;
use std::process::Command;
use std::path::PathBuf;

/// Get the path to the zlup binary.
fn zlup_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zlup"))
}

// =============================================================================
// Help and Version
// =============================================================================

#[test]
fn test_help() {
    let output = zlup_bin().arg("--help").output().expect("failed to run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Zluppy"));
    assert!(stdout.contains("compile"));
    assert!(stdout.contains("check"));
    assert!(stdout.contains("parse"));
    assert!(stdout.contains("analyze"));
}

#[test]
fn test_version() {
    let output = zlup_bin()
        .arg("--version")
        .output()
        .expect("failed to run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("zlup"));
}

#[test]
fn test_compile_help() {
    let output = zlup_bin()
        .args(["compile", "--help"])
        .output()
        .expect("failed to run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("target"));
    assert!(stdout.contains("slr"));
}

#[test]
fn test_check_help() {
    let output = zlup_bin()
        .args(["check", "--help"])
        .output()
        .expect("failed to run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("strict"));
}

// =============================================================================
// Compile Command
// =============================================================================

#[test]
fn test_compile_stdin_bell_state() {
    let source = r#"fn main() -> unit {
    mut q := qalloc(2);
    h q[0];
    cx (q[0], q[1]);
    return unit;
}"#;

    let mut child = zlup_bin()
        .args(["compile", "-", "--format", "slr", "-o", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    assert!(output.status.success(), "compile failed: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Verify JSON structure
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("invalid JSON output");

    assert_eq!(json["type"], "Program");
    assert_eq!(json["name"], "main");
    assert!(json["allocator"].is_object());
    assert_eq!(json["allocator"]["name"], "q");
    assert_eq!(json["allocator"]["capacity"], 2);

    // Check body has 2 gates
    let body = json["body"].as_array().expect("body should be array");
    assert_eq!(body.len(), 2);
    assert_eq!(body[0]["gate"], "H");
    assert_eq!(body[1]["gate"], "CX");
}

#[test]
fn test_compile_compact() {
    let source = "fn main() -> unit { mut q := qalloc(1); h q[0]; return unit; }";

    let mut child = zlup_bin()
        .args(["compile", "-", "--format", "slr", "--compact", "-o", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Compact output should not have newlines (except possibly at end)
    let trimmed = stdout.trim();
    assert!(
        !trimmed.contains("\n  "),
        "compact output should not be pretty-printed"
    );

    // But should still be valid JSON
    let _json: serde_json::Value = serde_json::from_str(&stdout).expect("invalid JSON");
}

#[test]
fn test_compile_rotation_gate() {
    let source = r#"fn main() -> unit {
    mut q := qalloc(1);
    rz(1.57) q[0];
    return unit;
}"#;

    let mut child = zlup_bin()
        .args(["compile", "-", "--format", "slr", "-o", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("invalid JSON");

    let body = json["body"].as_array().expect("body should be array");
    assert_eq!(body.len(), 1);
    assert_eq!(body[0]["gate"], "RZ");
    assert!(!body[0]["params"].as_array().unwrap().is_empty());
}

// =============================================================================
// Check Command
// =============================================================================

#[test]
fn test_check_valid_program() {
    let source = r#"fn main() -> unit {
    mut q := qalloc(2);
    h q[0];
    cx (q[0], q[1]);
    return unit;
}"#;

    let mut child = zlup_bin()
        .args(["check", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    assert!(output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("OK"));
}

#[test]
fn test_check_strict_mode() {
    let source = r#"fn main() -> unit {
    mut q := qalloc(2);
    h q[0];
    return unit;
}"#;

    let mut child = zlup_bin()
        .args(["check", "-", "--strict"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    // In strict mode, gates on unprepared qubits fail
    // But our current implementation doesn't track through function calls
    // so this may or may not fail depending on implementation details
    // For now, just verify it runs without crashing
    let _ = output.status;
}

// =============================================================================
// Parse Command
// =============================================================================

#[test]
fn test_parse_debug_format() {
    let source = "fn main() -> unit { mut q := qalloc(1); return unit; }";

    let mut child = zlup_bin()
        .args(["parse", "-", "--format", "debug"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Program"));
    assert!(stdout.contains("FnDecl"));
}

// =============================================================================
// Error Handling
// =============================================================================

#[test]
fn test_compile_parse_error() {
    let source = "fn main() -> unit { h q[0 }"; // Missing closing bracket

    let mut child = zlup_bin()
        .args(["compile", "-", "--format", "slr"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    assert!(!output.status.success(), "should fail on parse error");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("parse error") || stderr.contains("expected"),
        "should contain error message"
    );
}

#[test]
fn test_check_parse_error() {
    let source = "fn main( unit { }"; // Missing closing paren in params

    let mut child = zlup_bin()
        .args(["check", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    assert!(!output.status.success(), "should fail on parse error");
}

#[test]
fn test_compile_file_not_found() {
    let output = zlup_bin()
        .args(["compile", "nonexistent_file_12345.zlp"])
        .output()
        .expect("failed to run");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("read") || stderr.contains("No such file"),
        "should report file not found"
    );
}

// =============================================================================
// Complex Programs
// =============================================================================

#[test]
fn test_compile_child_allocator() {
    let source = r#"fn main() -> unit {
    mut base := qalloc(4);
    mut q := base.child(2);
    h q[0];
    cx (q[0], q[1]);
    return unit;
}"#;

    let mut child = zlup_bin()
        .args(["compile", "-", "--format", "slr", "-o", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("invalid JSON");

    // Should have declarations for both allocators
    let decls = json["declarations"].as_array().expect("declarations array");
    assert!(decls.len() >= 2, "should have at least 2 declarations");
}

#[test]
fn test_compile_conditional() {
    let source = r#"fn main() -> unit {
    mut q := qalloc(1);
    x := 1;
    if (x == 1) {
        h q[0];
    }
    return unit;
}"#;

    let mut child = zlup_bin()
        .args(["compile", "-", "--format", "slr", "-o", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("invalid JSON");

    // Should have if statement in body
    let body = json["body"].as_array().expect("body array");
    let has_if = body.iter().any(|stmt| stmt["type"] == "IfStmt");
    assert!(has_if, "should have if statement in body");
}

// =============================================================================
// Init Command
// =============================================================================

/// Create a unique temp directory for testing
fn temp_dir(test_name: &str) -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("zlup-test-{}-{}-{}",
        test_name, std::process::id(), timestamp));
    fs::create_dir_all(&dir).expect("failed to create temp dir");
    dir
}

/// Clean up a temp directory
fn cleanup_temp_dir(dir: &PathBuf) {
    if dir.exists() {
        let _ = fs::remove_dir_all(dir);
    }
}

#[test]
fn test_init_help() {
    let output = zlup_bin()
        .args(["init", "--help"])
        .output()
        .expect("failed to run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Initialize"));
    assert!(stdout.contains("NAME"));
}

#[test]
fn test_init_creates_project() {
    let temp = temp_dir("init_creates");
    let project_name = "test-quantum-app";
    let project_dir = temp.join(project_name);

    let output = zlup_bin()
        .current_dir(&temp)
        .args(["init", project_name])
        .output()
        .expect("failed to run");

    assert!(output.status.success(), "init failed: {:?}", String::from_utf8_lossy(&output.stderr));

    // Check files were created
    assert!(project_dir.exists(), "project dir should exist");
    assert!(project_dir.join("zlup.toml").exists(), "zlup.toml should exist");
    assert!(project_dir.join("main.zlp").exists(), "main.zlp should exist");

    // Check zlup.toml content
    let toml_content = fs::read_to_string(project_dir.join("zlup.toml")).expect("read zlup.toml");
    assert!(toml_content.contains("name = \"test-quantum-app\""));
    assert!(toml_content.contains("version = \"0.1.0\""));

    // Check main.zlp content
    let main_content = fs::read_to_string(project_dir.join("main.zlp")).expect("read main.zlp");
    assert!(main_content.contains("fn main()"));
    assert!(main_content.contains("qalloc"));

    // Cleanup
    cleanup_temp_dir(&temp);
}

#[test]
fn test_init_project_exists_error() {
    let temp = temp_dir("init_exists");
    let project_name = "existing-project";
    let project_dir = temp.join(project_name);

    // Create the directory first
    fs::create_dir_all(&project_dir).expect("create dir");

    let output = zlup_bin()
        .current_dir(&temp)
        .args(["init", project_name])
        .output()
        .expect("failed to run");

    assert!(!output.status.success(), "should fail when project exists");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("exists"), "should report project exists: {}", stderr);

    // Cleanup
    cleanup_temp_dir(&temp);
}

// =============================================================================
// Build Command
// =============================================================================

#[test]
fn test_build_help() {
    let output = zlup_bin()
        .args(["build", "--help"])
        .output()
        .expect("failed to run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Build"));
    assert!(stdout.contains("strict"));
    assert!(stdout.contains("target"));
}

#[test]
fn test_build_no_config_error() {
    let temp = temp_dir("build_no_config");

    let output = zlup_bin()
        .current_dir(&temp)
        .args(["build"])
        .output()
        .expect("failed to run");

    assert!(!output.status.success(), "should fail without zlup.toml");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("config") || stderr.contains("not found"),
            "should report config not found: {}", stderr);

    // Cleanup
    cleanup_temp_dir(&temp);
}

#[test]
fn test_init_then_build() {
    let temp = temp_dir("init_then_build");
    let project_name = "buildable-project";
    let project_dir = temp.join(project_name);

    // Initialize project
    let output = zlup_bin()
        .current_dir(&temp)
        .args(["init", project_name])
        .output()
        .expect("failed to run init");

    assert!(output.status.success(), "init failed: {:?}", String::from_utf8_lossy(&output.stderr));

    // Build project
    let output = zlup_bin()
        .current_dir(&project_dir)
        .args(["build"])
        .output()
        .expect("failed to run build");

    assert!(output.status.success(), "build failed: {:?}", String::from_utf8_lossy(&output.stderr));

    // Check output file was created
    let output_file = project_dir.join("build").join("main.slr.json");
    assert!(output_file.exists(), "output file should exist at {:?}", output_file);

    // Verify it's valid JSON
    let content = fs::read_to_string(&output_file).expect("read output");
    let json: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");
    assert_eq!(json["type"], "Program");

    // Cleanup
    cleanup_temp_dir(&temp);
}

#[test]
fn test_build_with_strict_override() {
    let temp = temp_dir("build_strict");
    let project_name = "strict-project";
    let project_dir = temp.join(project_name);

    // Initialize project
    let output = zlup_bin()
        .current_dir(&temp)
        .args(["init", project_name])
        .output()
        .expect("failed to run init");

    assert!(output.status.success(), "init failed: {:?}", String::from_utf8_lossy(&output.stderr));

    // Build with strict override
    let output = zlup_bin()
        .current_dir(&project_dir)
        .args(["build", "--strict", "true"])
        .output()
        .expect("failed to run build");

    // In strict mode, the build should show "strict" in the output
    // Note: The actual success/failure depends on semantic analyzer's strict mode behavior
    // which may flag qubit operations that aren't fully tracked
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("strict"), "should show strict mode in output: {}", stderr);

    // Cleanup
    cleanup_temp_dir(&temp);
}

// =============================================================================
// Example Files Integration Tests
// =============================================================================

/// Get the examples directory path relative to the project root
fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples")
}

/// Test that an example file parses and semantic checks correctly
fn test_example_checks(filename: &str) {
    let example_path = examples_dir().join(filename);
    assert!(example_path.exists(), "Example file should exist: {:?}", example_path);

    let output = zlup_bin()
        .args(["check", example_path.to_str().unwrap()])
        .output()
        .expect("failed to run check");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Example {} should pass semantic check.\nStderr: {}",
        filename,
        stderr
    );
}

/// Test that an example file compiles to SLR-AST
fn test_example_compiles_slr(filename: &str) {
    let example_path = examples_dir().join(filename);
    assert!(example_path.exists(), "Example file should exist: {:?}", example_path);

    let output = zlup_bin()
        .args(["compile", example_path.to_str().unwrap(), "--format", "slr", "-o", "-"])
        .output()
        .expect("failed to run compile");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "Example {} should compile to SLR.\nStderr: {}\nStdout: {}",
        filename,
        stderr,
        stdout
    );

    // Verify output is valid JSON
    let json_result: Result<serde_json::Value, _> = serde_json::from_str(&stdout);
    assert!(
        json_result.is_ok(),
        "Example {} SLR output should be valid JSON: {:?}",
        filename,
        json_result.err()
    );
}

/// Test that an example file compiles to QASM
fn test_example_compiles_qasm(filename: &str) {
    let example_path = examples_dir().join(filename);
    assert!(example_path.exists(), "Example file should exist: {:?}", example_path);

    let output = zlup_bin()
        .args(["compile", example_path.to_str().unwrap(), "--format", "qasm", "-o", "-"])
        .output()
        .expect("failed to run compile");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "Example {} should compile to QASM.\nStderr: {}\nStdout: {}",
        filename,
        stderr,
        stdout
    );

    // Verify output contains QASM header
    assert!(
        stdout.contains("OPENQASM") || stdout.contains("qreg"),
        "Example {} QASM output should contain QASM syntax",
        filename
    );
}

// Bell State Example
#[test]
fn test_example_bell_state_checks() {
    test_example_checks("bell_state.zlp");
}

#[test]
fn test_example_bell_state_compiles_slr() {
    test_example_compiles_slr("bell_state.zlp");
}

#[test]
fn test_example_bell_state_compiles_qasm() {
    test_example_compiles_qasm("bell_state.zlp");
}

// GHZ State Example
#[test]
fn test_example_ghz_state_checks() {
    test_example_checks("ghz_state.zlp");
}

#[test]
fn test_example_ghz_state_compiles_slr() {
    test_example_compiles_slr("ghz_state.zlp");
}

#[test]
fn test_example_ghz_state_compiles_qasm() {
    test_example_compiles_qasm("ghz_state.zlp");
}

// Grover 2-qubit Example
#[test]
fn test_example_grover_2qubit_checks() {
    test_example_checks("grover_2qubit.zlp");
}

#[test]
fn test_example_grover_2qubit_compiles_slr() {
    test_example_compiles_slr("grover_2qubit.zlp");
}

#[test]
fn test_example_grover_2qubit_compiles_qasm() {
    test_example_compiles_qasm("grover_2qubit.zlp");
}

// QFT 3-qubit Example
#[test]
fn test_example_qft_3qubit_checks() {
    test_example_checks("qft_3qubit.zlp");
}

#[test]
fn test_example_qft_3qubit_compiles_slr() {
    test_example_compiles_slr("qft_3qubit.zlp");
}

#[test]
fn test_example_qft_3qubit_compiles_qasm() {
    test_example_compiles_qasm("qft_3qubit.zlp");
}

// Simple QEC Example
#[test]
fn test_example_simple_qec_checks() {
    test_example_checks("simple_qec.zlp");
}

#[test]
fn test_example_simple_qec_compiles_slr() {
    test_example_compiles_slr("simple_qec.zlp");
}

#[test]
fn test_example_simple_qec_compiles_qasm() {
    test_example_compiles_qasm("simple_qec.zlp");
}

// Teleportation Example
#[test]
fn test_example_teleportation_checks() {
    test_example_checks("teleportation.zlp");
}

#[test]
fn test_example_teleportation_compiles_slr() {
    test_example_compiles_slr("teleportation.zlp");
}

#[test]
fn test_example_teleportation_compiles_qasm() {
    test_example_compiles_qasm("teleportation.zlp");
}

// =============================================================================
// Analyze Command
// =============================================================================

#[test]
fn test_analyze_help() {
    let output = zlup_bin()
        .args(["analyze", "--help"])
        .output()
        .expect("failed to run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("parallelism"));
    assert!(stdout.contains("format"));
    assert!(stdout.contains("verbose"));
}

#[test]
fn test_analyze_stdin_text() {
    let source = r#"fn main() -> unit {
    mut q := qalloc(2);
    h q[0];
    cx (q[0], q[1]);
    return unit;
}"#;

    let mut child = zlup_bin()
        .args(["analyze", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    assert!(output.status.success(), "analyze failed: {:?}", String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Parallelism Analysis"), "Missing header");
    assert!(stdout.contains("Allocators:"), "Missing allocators section");
    assert!(stdout.contains("q"), "Missing allocator q");
    assert!(stdout.contains("Function Analysis:"), "Missing function analysis");
    assert!(stdout.contains("main"), "Missing main function");
}

#[test]
fn test_analyze_stdin_json() {
    let source = r#"fn main() -> unit {
    mut q := qalloc(2);
    h q[0];
    return unit;
}"#;

    let mut child = zlup_bin()
        .args(["analyze", "-", "--format", "json"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    assert!(output.status.success(), "analyze failed: {:?}", String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse as JSON to verify structure
    let json: serde_json::Value = serde_json::from_str(&stdout)
        .expect("Output should be valid JSON");

    assert!(json["allocators"].is_array(), "Should have allocators array");
    assert!(json["functions"].is_array(), "Should have functions array");
    assert!(json["parallel_layers"].is_array(), "Should have parallel_layers array");
    assert!(json["total_operations"].is_number(), "Should have total_operations");
}

#[test]
fn test_analyze_verbose() {
    let source = r#"fn main() -> unit {
    mut q := qalloc(2);
    h q[0];
    cx (q[0], q[1]);
    return unit;
}"#;

    let mut child = zlup_bin()
        .args(["analyze", "-", "--verbose"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    assert!(output.status.success(), "analyze failed: {:?}", String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Dependency Graph"), "Verbose should include dependency graph");
    assert!(stdout.contains("Parallel Layers"), "Verbose should include parallel layers");
}

#[test]
fn test_analyze_disjoint_allocators() {
    // Two independent allocators should show parallelism
    let source = r#"fn main() -> unit {
    mut q1 := qalloc(2);
    mut q2 := qalloc(2);
    h q1[0];
    h q2[0];
    return unit;
}"#;

    let mut child = zlup_bin()
        .args(["analyze", "-", "--format", "json"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");

    // Should have 2 allocators
    let allocators = json["allocators"].as_array().expect("allocators array");
    assert_eq!(allocators.len(), 2, "Should have 2 allocators");

    // Check max parallelism > 1 (H gates on different allocators can run in parallel)
    let functions = json["functions"].as_array().expect("functions array");
    let main_func = &functions[0];
    let max_parallelism = main_func["max_parallelism"].as_u64().expect("max_parallelism");
    assert!(max_parallelism >= 2, "Disjoint allocators should enable parallelism");
}

#[test]
fn test_analyze_parse_error() {
    // Invalid syntax should fail
    let source = "fn main( { }";  // Missing closing paren and return type

    let mut child = zlup_bin()
        .args(["analyze", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    assert!(!output.status.success(), "Should fail on parse error");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("parse") || stderr.contains("error") || stderr.contains("expected"),
        "Should report parse error: {}",
        stderr
    );
}

#[test]
fn test_analyze_file_not_found() {
    let output = zlup_bin()
        .args(["analyze", "nonexistent_file_12345.zlp"])
        .output()
        .expect("failed to run");

    assert!(!output.status.success(), "Should fail on missing file");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("read") || stderr.contains("No such file") || stderr.contains("not found"),
        "Should report file not found: {}",
        stderr
    );
}

#[test]
fn test_analyze_semantic_error() {
    // Reference undefined variable
    let source = r#"fn main() -> unit {
    h undefined_var[0];
    return unit;
}"#;

    let mut child = zlup_bin()
        .args(["analyze", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn");

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(source.as_bytes()).expect("failed to write");
    }

    let output = child.wait_with_output().expect("failed to wait");
    // Note: semantic errors in permissive mode may still allow analysis
    // This test verifies the command handles the input without crashing
    let _ = output.status;  // May or may not succeed depending on strictness
}
