//! Implementation of the `rust` subcommand (CUDA-aware cargo commands)

use pecos_build::Result;
use pecos_build::errors::Error;
use serde_json::Value;
use std::process::Command;

/// FFI crates that should be excluded from workspace-wide cargo commands
const FFI_CRATES: &[&str] = &["pecos-rslib", "pecos-julia-ffi", "pecos-go-ffi"];

/// Warn if shared C++ dependencies differ across per-crate pecos.toml files.
/// This is informational -- different crates may legitimately pin different versions.
fn check_dep_consistency() {
    let Ok(workspace_root) = std::env::current_dir() else {
        return;
    };
    let Ok(mismatches) = pecos_build::manifest::check_consistency(&workspace_root) else {
        return;
    };
    if mismatches.is_empty() {
        return;
    }
    eprintln!("Note: some C++ dependencies differ across per-crate pecos.toml files:");
    for m in &mismatches {
        eprintln!("  {} ({}): {:?}", m.dep_name, m.field, m.values);
    }
    eprintln!();
}

#[derive(Debug)]
enum GpuProbeResult {
    Available,
    Unavailable,
    ProbeFailed(String),
}

/// Run the rust subcommand
pub fn run(command: &super::RustCommands) -> Result<()> {
    match command {
        super::RustCommands::Check { include_ffi } => run_check(*include_ffi),
        super::RustCommands::Clippy { include_ffi, fix } => run_clippy(*include_ffi, *fix),
        super::RustCommands::Test {
            release,
            include_ffi,
        } => run_test(*release, *include_ffi),
    }
}

/// Probe whether cuQuantum is available at runtime (SDK installed + CUDA GPU present).
///
/// Runs the cuda-check binary from pecos-cuquantum-sys which tries to load the
/// cuQuantum shared libraries and call cudaDeviceSynchronize. This catches the case
/// where the SDK is installed for compilation but no GPU is present to run code.
fn probe_cuquantum_availability() -> bool {
    let output = Command::new("cargo")
        .args([
            "run",
            "-p",
            "pecos-cuquantum-sys",
            "--bin",
            "cuda_check",
            "-q",
            "--",
            "--json",
        ])
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Ok(payload) = serde_json::from_str::<Value>(stdout.trim()) {
                return payload.get("status").and_then(Value::as_str) == Some("available");
            }
            true
        }
        _ => false,
    }
}

/// Probe whether a GPU adapter is available for wgpu.
///
/// Runs the gpu-check binary from pecos-gpu-sims. A clean non-zero exit with no
/// output is treated as "no GPU available". If cargo emits diagnostics, we
/// preserve that separately so higher-level commands can avoid silently
/// misclassifying a probe/build failure as "no GPU".
fn probe_gpu_availability() -> GpuProbeResult {
    let output = Command::new("cargo")
        .args([
            "run",
            "-p",
            "pecos-gpu-sims",
            "--bin",
            "gpu-check",
            "-q",
            "--",
            "--json",
        ])
        .output();

    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if let Ok(payload) = serde_json::from_str::<Value>(&stdout) {
                let status = payload.get("status").and_then(Value::as_str);
                let message = payload
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("gpu-check did not provide a message");
                return match status {
                    Some("available") if output.status.success() => GpuProbeResult::Available,
                    Some("unavailable") => GpuProbeResult::Unavailable,
                    Some(other) => GpuProbeResult::ProbeFailed(format!("{other}: {message}")),
                    None => GpuProbeResult::ProbeFailed(
                        "gpu-check JSON did not include a status".to_string(),
                    ),
                };
            }

            let details = [stderr, stdout]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n");

            if details.is_empty() {
                match output.status.code() {
                    Some(1) => GpuProbeResult::Unavailable,
                    Some(code) => {
                        GpuProbeResult::ProbeFailed(format!("gpu-check exited with code {code}"))
                    }
                    None => GpuProbeResult::ProbeFailed(
                        "gpu-check terminated without an exit code".to_string(),
                    ),
                }
            } else {
                GpuProbeResult::ProbeFailed(details)
            }
        }
        Err(error) => GpuProbeResult::ProbeFailed(format!("Failed to run gpu-check: {error}")),
    }
}

fn should_include_gpu_sims(gpu_probe: &GpuProbeResult) -> bool {
    matches!(gpu_probe, GpuProbeResult::Available)
}

fn maybe_print_gpu_probe_status(gpu_probe: &GpuProbeResult, include_gpu_sims: bool) {
    if include_gpu_sims {
        if matches!(gpu_probe, GpuProbeResult::Available) {
            println!("GPU probe succeeded - including pecos-gpu-sims");
        }
    } else {
        match gpu_probe {
            GpuProbeResult::Unavailable => {
                println!("GPU not detected - excluding pecos-gpu-sims");
            }
            GpuProbeResult::ProbeFailed(details) => {
                println!("GPU probe failed - excluding pecos-gpu-sims:\n{details}");
            }
            GpuProbeResult::Available => {}
        }
    }
}

/// Check if a tool is available
fn is_tool_available(tool: &str) -> bool {
    Command::new(tool)
        .args(["--version"])
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Run a cargo command and return success status.
///
/// Applies the PECOS build environment (`CMAKE`, `LLVM_SYS_140_PREFIX`,
/// `SDKROOT`, etc.) so build scripts like highs-sys's cmake-rs invocation
/// find the PECOS-managed cmake without further plumbing.
fn run_cargo_command(args: &[&str]) -> bool {
    let mut cmd = Command::new("cargo");
    cmd.args(args);
    for (key, value) in super::env_cmd::collect_env() {
        cmd.env(key, value);
    }
    matches!(cmd.status(), Ok(s) if s.success())
}

/// Run cargo check with GPU-aware feature handling
#[allow(clippy::too_many_lines)]
fn run_check(include_ffi: bool) -> Result<()> {
    let gpu_probe = probe_gpu_availability();
    let include_gpu_sims = should_include_gpu_sims(&gpu_probe);

    maybe_print_gpu_probe_status(&gpu_probe, include_gpu_sims);

    println!("Checking workspace packages...");
    let mut args: Vec<&str> = vec![
        "check",
        "--workspace",
        "--all-targets",
        "--all-features",
        "--exclude=pecos-cuquantum", // Requires cuQuantum SDK
    ];

    let mut exclude_flags: Vec<String> = Vec::new();
    if !include_ffi {
        exclude_flags.extend(FFI_CRATES.iter().map(|c| format!("--exclude={c}")));
    }
    if !include_gpu_sims {
        exclude_flags.push("--exclude=pecos-gpu-sims".to_string());
    }
    for flag in &exclude_flags {
        args.push(flag);
    }

    if !run_cargo_command(&args) {
        return Err(Error::Config("cargo check failed".to_string()));
    }

    if include_ffi {
        println!("Checking pecos-rslib...");
        if !run_cargo_command(&[
            "check",
            "-p",
            "pecos-rslib",
            "--all-targets",
            "--all-features",
        ]) {
            return Err(Error::Config(
                "cargo check (pecos-rslib) failed".to_string(),
            ));
        }

        if is_tool_available("julia") {
            println!("Checking pecos-julia-ffi...");
            if !run_cargo_command(&[
                "check",
                "-p",
                "pecos-julia-ffi",
                "--all-targets",
                "--all-features",
            ]) {
                return Err(Error::Config(
                    "cargo check (pecos-julia-ffi) failed".to_string(),
                ));
            }
        }

        if is_tool_available("go") {
            println!("Checking pecos-go-ffi...");
            if !run_cargo_command(&[
                "check",
                "-p",
                "pecos-go-ffi",
                "--all-targets",
                "--all-features",
            ]) {
                return Err(Error::Config(
                    "cargo check (pecos-go-ffi) failed".to_string(),
                ));
            }
        }
    }

    println!();
    println!("cargo check completed successfully");
    Ok(())
}

/// Run cargo clippy with GPU-aware feature handling
#[allow(clippy::too_many_lines)]
fn run_clippy(include_ffi: bool, fix: bool) -> Result<()> {
    let gpu_probe = probe_gpu_availability();
    let include_gpu_sims = should_include_gpu_sims(&gpu_probe);

    maybe_print_gpu_probe_status(&gpu_probe, include_gpu_sims);

    let fix_args: Vec<&str> = if fix {
        vec!["--fix", "--allow-staged", "--allow-dirty"]
    } else {
        vec![]
    };

    println!("Running clippy on workspace packages...");
    let mut args: Vec<&str> = vec![
        "clippy",
        "--workspace",
        "--all-targets",
        "--all-features",
        "--exclude=pecos-cuquantum", // Requires cuQuantum SDK
    ];
    args.extend(&fix_args);

    let mut exclude_flags: Vec<String> = Vec::new();
    if !include_ffi {
        exclude_flags.extend(FFI_CRATES.iter().map(|c| format!("--exclude={c}")));
    }
    if !include_gpu_sims {
        exclude_flags.push("--exclude=pecos-gpu-sims".to_string());
    }
    for flag in &exclude_flags {
        args.push(flag);
    }

    args.extend(&["--", "-D", "warnings"]);

    if !run_cargo_command(&args) {
        return Err(Error::Config("cargo clippy failed".to_string()));
    }

    if include_ffi {
        println!("Running clippy on pecos-rslib...");
        let mut args: Vec<&str> = vec![
            "clippy",
            "-p",
            "pecos-rslib",
            "--all-targets",
            "--all-features",
        ];
        args.extend(&fix_args);
        args.extend(&["--", "-D", "warnings"]);
        if !run_cargo_command(&args) {
            return Err(Error::Config(
                "cargo clippy (pecos-rslib) failed".to_string(),
            ));
        }

        if is_tool_available("julia") {
            println!("Running clippy on pecos-julia-ffi...");
            let mut args: Vec<&str> = vec![
                "clippy",
                "-p",
                "pecos-julia-ffi",
                "--all-targets",
                "--all-features",
            ];
            args.extend(&fix_args);
            args.extend(&["--", "-D", "warnings"]);
            if !run_cargo_command(&args) {
                return Err(Error::Config(
                    "cargo clippy (pecos-julia-ffi) failed".to_string(),
                ));
            }
        }

        if is_tool_available("go") {
            println!("Running clippy on pecos-go-ffi...");
            let mut args: Vec<&str> = vec![
                "clippy",
                "-p",
                "pecos-go-ffi",
                "--all-targets",
                "--all-features",
            ];
            args.extend(&fix_args);
            args.extend(&["--", "-D", "warnings"]);
            if !run_cargo_command(&args) {
                return Err(Error::Config(
                    "cargo clippy (pecos-go-ffi) failed".to_string(),
                ));
            }
        }
    }

    println!();
    println!("cargo clippy completed successfully");
    Ok(())
}

/// Run cargo test with GPU-aware feature handling
fn run_test(release: bool, include_ffi: bool) -> Result<()> {
    // Warn about any C++ dependency version differences across crates
    check_dep_consistency();

    let gpu_probe = probe_gpu_availability();
    let include_gpu_sims = should_include_gpu_sims(&gpu_probe);
    let release_flag = if release { "--release" } else { "" };

    maybe_print_gpu_probe_status(&gpu_probe, include_gpu_sims);

    println!("Testing workspace packages...");
    // runtime = sim + qasm + phir (format parsers)
    // hugr = qis (includes llvm) + hugr compilation
    // pecos-cli is excluded here and tested separately below with --features=runtime
    // to ensure the pecos binary has PHIR/QIS support for integration tests.
    let mut args: Vec<&str> = vec!["test", "--workspace", "--features=runtime,hugr"];

    for crate_name in FFI_CRATES {
        args.push("--exclude");
        args.push(crate_name);
    }

    args.extend(&[
        "--exclude",
        "pecos-cuquantum", // Requires cuQuantum SDK, test separately if available
        "--exclude",
        "pecos-cli", // Test separately with --features=runtime (see below)
        "--exclude",
        "pecos-decoders",
        "--exclude",
        "pecos-gpu-sims", // Always exclude from workspace test, test separately if GPU available
    ]);

    if !release_flag.is_empty() {
        args.push(release_flag);
    }

    if !run_cargo_command(&args) {
        return Err(Error::Config("cargo test (workspace) failed".to_string()));
    }

    // Test pecos-cli separately with --features=runtime.
    // cargo test --workspace --features=runtime overwrites the pecos binary
    // WITHOUT runtime features (cargo feature unification bug), so the CLI
    // integration tests that invoke `cargo_bin!("pecos")` would get a broken
    // binary. Testing separately ensures the binary is built correctly.
    println!("Testing pecos-cli with runtime features...");
    let mut cli_args: Vec<&str> = vec!["test", "-p", "pecos-cli", "--features=runtime"];
    if !release_flag.is_empty() {
        cli_args.push(release_flag);
    }
    if !run_cargo_command(&cli_args) {
        return Err(Error::Config(
            "cargo test (pecos-cli with runtime) failed".to_string(),
        ));
    }

    // Test cuQuantum if SDK is available (requires both CUDA and cuQuantum)
    if probe_cuquantum_availability() {
        println!("cuQuantum runtime available - testing pecos-cuquantum");
        let mut args = vec!["test", "-p", "pecos-cuquantum"];
        if !release_flag.is_empty() {
            args.push(release_flag);
        }
        if !run_cargo_command(&args) {
            return Err(Error::Config(
                "cargo test (pecos-cuquantum) failed".to_string(),
            ));
        }
    } else {
        println!("cuQuantum runtime not available - skipping pecos-cuquantum");
    }

    if include_gpu_sims {
        println!("Including pecos-gpu-sims in Rust tests");
        let mut args = vec!["test", "-p", "pecos-gpu-sims"];
        if !release_flag.is_empty() {
            args.push(release_flag);
        }
        if !run_cargo_command(&args) {
            return Err(Error::Config(
                "cargo test (pecos-gpu-sims) failed".to_string(),
            ));
        }
    }

    println!("Testing pecos-decoders...");
    let mut args = vec!["test", "-p", "pecos-decoders", "--all-features"];
    if !release_flag.is_empty() {
        args.push(release_flag);
    }
    if !run_cargo_command(&args) {
        return Err(Error::Config(
            "cargo test (pecos-decoders) failed".to_string(),
        ));
    }

    if include_ffi {
        println!("Testing pecos-rslib...");
        let mut args = vec!["test", "-p", "pecos-rslib", "--all-features"];
        if !release_flag.is_empty() {
            args.push(release_flag);
        }
        if !run_cargo_command(&args) {
            return Err(Error::Config("cargo test (pecos-rslib) failed".to_string()));
        }
    }

    println!();
    println!("cargo test completed successfully");
    Ok(())
}
