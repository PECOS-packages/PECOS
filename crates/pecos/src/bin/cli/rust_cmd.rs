//! Implementation of the `rust` subcommand (CUDA-aware cargo commands)

use cargo_metadata::MetadataCommand;
use pecos_build::Result;
use pecos_build::errors::Error;
use serde_json::Value;
use std::collections::BTreeSet;
use std::process::Command;

/// FFI crates that should be excluded from workspace-wide cargo commands
const FFI_CRATES: &[&str] = &["pecos-rslib", "pecos-julia-ffi", "pecos-go-ffi"];

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
        super::RustCommands::Fmt { check } => run_fmt(*check),
        super::RustCommands::Bench {
            profile,
            features,
            pattern,
        } => run_bench(profile, features.as_deref(), pattern.as_deref()),
    }
}

/// Check if CUDA is available (local ~/.pecos/cuda/ or system)
fn is_cuda_available() -> bool {
    pecos_build::cuda::is_cuda_available()
}

/// Check if cuQuantum SDK is available
fn is_cuquantum_available() -> bool {
    pecos_build::cuquantum::is_cuquantum_available()
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
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Get features for a package excluding certain features
///
/// Uses `cargo_metadata` directly instead of spawning a subprocess to avoid
/// Windows file locking issues (running executable can't be replaced).
fn get_features_excluding(package: &str, exclude: &str) -> Result<String> {
    let metadata = MetadataCommand::new()
        .no_deps()
        .exec()
        .map_err(|e| Error::Config(format!("Failed to get cargo metadata: {e}")))?;

    let pkg = metadata
        .packages
        .iter()
        .find(|p| p.name == package)
        .ok_or_else(|| Error::Config(format!("Package '{package}' not found in workspace")))?;

    let exclusions: BTreeSet<&str> = exclude.split(',').map(str::trim).collect();

    let features: Vec<&str> = pkg
        .features
        .keys()
        .map(String::as_str)
        .filter(|f| !exclusions.contains(f))
        .collect();

    Ok(features.join(","))
}

/// Run a cargo command and return success status
fn run_cargo_command(args: &[&str]) -> bool {
    let status = Command::new("cargo").args(args).status();
    matches!(status, Ok(s) if s.success())
}

/// Run cargo check with CUDA-aware and GPU-aware feature handling
#[allow(clippy::too_many_lines)]
fn run_check(include_ffi: bool) -> Result<()> {
    let cuda_available = is_cuda_available();
    let gpu_probe = probe_gpu_availability();
    let include_gpu_sims = should_include_gpu_sims(&gpu_probe);

    maybe_print_gpu_probe_status(&gpu_probe, include_gpu_sims);

    if cuda_available {
        println!("CUDA detected - checking with all features");

        let mut args: Vec<&str> = vec!["check", "--workspace", "--all-targets", "--all-features"];

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
    } else {
        println!("CUDA not detected - checking all features except CUDA");

        println!(
            "Checking workspace packages (excluding FFI crates and those with CUDA features)..."
        );
        let mut args: Vec<&str> = vec![
            "check",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--exclude=pecos",
            "--exclude=pecos-quest",
            "--exclude=pecos-cuquantum", // Requires cuQuantum SDK
            // benchmarks depends on pecos, and --all-features enables pecos/cuda
            "--exclude=benchmarks",
        ];

        let mut exclude_flags: Vec<String> = FFI_CRATES
            .iter()
            .map(|c| format!("--exclude={c}"))
            .collect();
        if !include_gpu_sims {
            exclude_flags.push("--exclude=pecos-gpu-sims".to_string());
        }
        for flag in &exclude_flags {
            args.push(flag);
        }

        if !run_cargo_command(&args) {
            return Err(Error::Config("cargo check (workspace) failed".to_string()));
        }

        println!("Checking pecos with all features except cuda...");
        let pecos_features = get_features_excluding("pecos", "cuda")?;
        let features_arg = format!("--features={pecos_features}");
        if !run_cargo_command(&["check", "-p", "pecos", "--all-targets", &features_arg]) {
            return Err(Error::Config("cargo check (pecos) failed".to_string()));
        }

        println!("Checking pecos-quest with all features except cuda...");
        let quest_features = get_features_excluding("pecos-quest", "cuda")?;
        let features_arg = format!("--features={quest_features}");
        if !run_cargo_command(&["check", "-p", "pecos-quest", "--all-targets", &features_arg]) {
            return Err(Error::Config(
                "cargo check (pecos-quest) failed".to_string(),
            ));
        }
    }

    if include_ffi {
        println!("Checking pecos-rslib...");
        // Only use --all-features if CUDA is available, otherwise exclude cuda
        if cuda_available {
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
        } else {
            let rslib_features = get_features_excluding("pecos-rslib", "cuda")?;
            let features_arg = format!("--features={rslib_features}");
            if !run_cargo_command(&["check", "-p", "pecos-rslib", "--all-targets", &features_arg]) {
                return Err(Error::Config(
                    "cargo check (pecos-rslib) failed".to_string(),
                ));
            }
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

/// Run cargo clippy with CUDA-aware and GPU-aware feature handling
#[allow(clippy::too_many_lines)]
fn run_clippy(include_ffi: bool, fix: bool) -> Result<()> {
    let cuda_available = is_cuda_available();
    let gpu_probe = probe_gpu_availability();
    let include_gpu_sims = should_include_gpu_sims(&gpu_probe);

    maybe_print_gpu_probe_status(&gpu_probe, include_gpu_sims);

    let fix_args: Vec<&str> = if fix {
        vec!["--fix", "--allow-staged", "--allow-dirty"]
    } else {
        vec![]
    };

    if cuda_available {
        println!("CUDA detected - running clippy with all features");

        let mut args: Vec<&str> = vec!["clippy", "--workspace", "--all-targets", "--all-features"];
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
    } else {
        println!("CUDA not detected - running clippy on all features except CUDA");

        println!(
            "Running clippy on workspace packages (excluding FFI crates and those with CUDA features)..."
        );
        let mut args: Vec<&str> = vec![
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--exclude=pecos",
            "--exclude=pecos-quest",
            "--exclude=pecos-cuquantum", // Requires cuQuantum SDK
            // benchmarks depends on pecos, and --all-features enables pecos/cuda
            "--exclude=benchmarks",
        ];
        args.extend(&fix_args);

        let mut exclude_flags: Vec<String> = FFI_CRATES
            .iter()
            .map(|c| format!("--exclude={c}"))
            .collect();
        if !include_gpu_sims {
            exclude_flags.push("--exclude=pecos-gpu-sims".to_string());
        }
        for flag in &exclude_flags {
            args.push(flag);
        }

        args.extend(&["--", "-D", "warnings"]);

        if !run_cargo_command(&args) {
            return Err(Error::Config("cargo clippy (workspace) failed".to_string()));
        }

        println!("Running clippy on pecos with all features except cuda...");
        let pecos_features = get_features_excluding("pecos", "cuda")?;
        let features_arg = format!("--features={pecos_features}");
        let mut args: Vec<&str> = vec!["clippy", "-p", "pecos", "--all-targets", &features_arg];
        args.extend(&fix_args);
        args.extend(&["--", "-D", "warnings"]);
        if !run_cargo_command(&args) {
            return Err(Error::Config("cargo clippy (pecos) failed".to_string()));
        }

        println!("Running clippy on pecos-quest with all features except cuda...");
        let quest_features = get_features_excluding("pecos-quest", "cuda")?;
        let features_arg = format!("--features={quest_features}");
        let mut args: Vec<&str> = vec![
            "clippy",
            "-p",
            "pecos-quest",
            "--all-targets",
            &features_arg,
        ];
        args.extend(&fix_args);
        args.extend(&["--", "-D", "warnings"]);
        if !run_cargo_command(&args) {
            return Err(Error::Config(
                "cargo clippy (pecos-quest) failed".to_string(),
            ));
        }
    }

    if include_ffi {
        println!("Running clippy on pecos-rslib...");
        let mut args: Vec<&str> = vec!["clippy", "-p", "pecos-rslib", "--all-targets"];
        // Only use --all-features if CUDA is available, otherwise exclude cuda
        if cuda_available {
            args.push("--all-features");
        } else {
            let rslib_features = get_features_excluding("pecos-rslib", "cuda")?;
            let features_arg_owned = format!("--features={rslib_features}");
            // Need to leak the string to get a &'static str for the args vec
            let features_arg: &'static str = Box::leak(features_arg_owned.into_boxed_str());
            args.push(features_arg);
        }
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

/// Run cargo test with CUDA-aware and GPU-aware feature handling
fn run_test(release: bool, include_ffi: bool) -> Result<()> {
    let cuda_available = is_cuda_available();
    let gpu_probe = probe_gpu_availability();
    let include_gpu_sims = should_include_gpu_sims(&gpu_probe);
    let release_flag = if release { "--release" } else { "" };

    maybe_print_gpu_probe_status(&gpu_probe, include_gpu_sims);

    println!("Testing workspace packages...");
    // runtime = sim + qasm + phir (format parsers)
    // hugr = qis (includes llvm) + hugr compilation
    let mut args: Vec<&str> = vec!["test", "--workspace", "--features=runtime,hugr"];

    for crate_name in FFI_CRATES {
        args.push("--exclude");
        args.push(crate_name);
    }

    args.extend(&[
        "--exclude",
        "pecos-quest",
        "--exclude",
        "pecos-cuquantum", // Requires cuQuantum SDK, test separately if CUDA available
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

    if cuda_available {
        println!("CUDA detected - testing pecos-quest with all features");
        let mut args = vec!["test", "-p", "pecos-quest", "--all-features"];
        if !release_flag.is_empty() {
            args.push(release_flag);
        }
        if !run_cargo_command(&args) {
            return Err(Error::Config("cargo test (pecos-quest) failed".to_string()));
        }
    } else {
        println!("CUDA not detected - testing pecos-quest with cpu features only");
        let mut args = vec!["test", "-p", "pecos-quest", "--features=cpu"];
        if !release_flag.is_empty() {
            args.push(release_flag);
        }
        if !run_cargo_command(&args) {
            return Err(Error::Config("cargo test (pecos-quest) failed".to_string()));
        }
    }

    // Test cuQuantum if SDK is available (requires both CUDA and cuQuantum)
    if is_cuquantum_available() {
        println!("cuQuantum SDK detected - testing pecos-cuquantum");
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
        println!("cuQuantum SDK not detected - skipping pecos-cuquantum");
    }

    // Test GPU simulator if GPU is available
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

/// Run cargo fmt
fn run_fmt(check: bool) -> Result<()> {
    let mut args = vec!["fmt", "--all"];
    if check {
        args.extend(&["--", "--check"]);
    }

    if !run_cargo_command(&args) {
        if check {
            return Err(Error::Config(
                "cargo fmt check failed - formatting issues found".to_string(),
            ));
        }
        return Err(Error::Config("cargo fmt failed".to_string()));
    }

    if check {
        println!("All Rust code is properly formatted");
    } else {
        println!("Rust code formatted successfully");
    }
    Ok(())
}

/// Run cargo bench with configurable profile and features
fn run_bench(profile: &str, features: Option<&str>, pattern: Option<&str>) -> Result<()> {
    let mut cmd = Command::new("cargo");
    cmd.args(["bench", "-p", "benchmarks", "--bench", "benchmarks"]);

    match profile {
        "native" => {
            println!("Running benchmarks with native CPU optimizations...");
            cmd.arg("--profile=native");
            // Preserve any existing RUSTFLAGS while adding target-cpu=native
            let mut rustflags = std::env::var("RUSTFLAGS").unwrap_or_default();
            if !rustflags.is_empty() {
                rustflags.push(' ');
            }
            rustflags.push_str("-C target-cpu=native");
            cmd.env("RUSTFLAGS", rustflags);
        }
        "release" => {
            println!("Running benchmarks in release mode...");
        }
        other => {
            return Err(Error::Config(format!(
                "Unknown bench profile '{other}'. Use 'release' or 'native'."
            )));
        }
    }

    if let Some(feat) = features {
        cmd.arg(format!("--features={feat}"));
    }

    if let Some(pat) = pattern {
        cmd.args(["--", pat]);
    }

    let status = cmd.status();
    if !matches!(status, Ok(s) if s.success()) {
        return Err(Error::Config("cargo bench failed".to_string()));
    }

    println!();
    println!("Benchmarks completed successfully");
    Ok(())
}
